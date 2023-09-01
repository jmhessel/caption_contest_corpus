'''
Fine-tunes causal LMs like llama2. Some features supported:

- qlora/lora/full finetune. auto-gptq support is WIP due to numerical instability (see clamp_llamarmsnorm_forward)
- accelerate
- prompt vs. completion loss
- gradient accumulation / gradient checkpointing

Here's an example command to train a joke explanation model:

accelerate launch train.py datasets/train_joke_explain.jsonl datasets/val_joke_explain.jsonl explanation_generation --model meta-llama/Llama-2-$size\-hf --batch_size 4 --lr $lr --generate_during_val 0 --n_epochs 5 --use_lora 1 --load_in_4bit 1 --gradient_checkpointing 1

'''
import argparse
import collections
import tqdm
import json
import numpy as np
import os
import transformers
import accelerate
import tempfile
import torch
import subprocess
import pprint
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train')
    parser.add_argument('val')
    parser.add_argument('task_name')

    parser.add_argument('--model',
                        default='meta-llama/Llama-2-7b-hf',
                        type=str)

    parser.add_argument('--tokenizer',
                        default=None,
                        help='if the tokenizer and the model are saved separately, then use this to specify just the tokenizer',
                        type=str)

    parser.add_argument('--load_in_4bit',
                        default=0,
                        type=int,
                        help='should we load bitsnbytes 4 bit?')

    parser.add_argument('--prompt_loss_weight',
                        default=0.1,
                        help='tradeoff between prompt modeling/completion modeling',
                        type=float)

    parser.add_argument('--batch_size',
                        default=32,
                        type=int)

    parser.add_argument('--n_epochs',
                        type=int,
                        default=10)

    parser.add_argument('--lr',
                        type=float,
                        default=.00001)

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help='how many steps for gradient accumulation')

    parser.add_argument('--skip_save',
                        type=int,
                        default=0)

    parser.add_argument('--force_run',
                        type=int,
                        default=0,
                        help='if 1, we will force the run even if the output already exists.')

    parser.add_argument('--use_lora',
                        type=int,
                        default=0,
                        help='if 1, we will use LoRA')

    parser.add_argument('--lora_r',
                        type=int,
                        default=32,
                        help='what rank LoRA should be used?')

    parser.add_argument('--just_val',
                        type=int,
                        default=0,
                        help='if 1 we will just do val. good for debugging.')

    parser.add_argument('--gradient_checkpointing',
                        type=int,
                        default=1,
                        help='if 1 we will use gradient checkpointing which is slower, but better memory')

    parser.add_argument('--generate_during_val',
                        type=int,
                        default=1,
                        help='if 1 we will generate actual samples during validation and print them out.')

    parser.add_argument('--use_bfloat',
                        type=int,
                        default=0,
                        help='if 1 we will use bfloat16 instead of float16 for quantized models')

    parser.add_argument('--use_adafactor',
                        type=int,
                        default=0,
                        help='if 1 we will use adafactor instead of adamw')

    args = parser.parse_args()
    args.val_stat = 'loss'

    if not args.tokenizer:
        args.tokenizer = args.model

    args.output_path = (args.task_name + '~val{}'.format(args.val_stat) +
                        '={:.5f}' + '~model=' + '{}'.format(args.model.replace('/', '+')) +
                        '~lora={}'.format(args.use_lora if args.use_lora == 0 else args.lora_r) + '~lr={}'.format(args.lr) +
                        '~4bit={}'.format(args.load_in_4bit) + '~promptloss={}'.format(args.prompt_loss_weight) +
                        ('.pt' if not args.use_lora else ''))
    if not args.force_run:
        toks = args.output_path.split('/')
        outdir = '/'.join(toks[:-1]) if len(toks) > 1 else '.'
        def fnameparse(x):
            return (x.split('~val')[0], '~'.join(x.split('~val')[1].split('~')[1:]))
        if fnameparse(args.output_path) in set([fnameparse(x) for x in os.listdir(outdir) if '~model=' in x]):
            print('{} done already, run with --force_run to run.'.format(args.output_path))
            quit()

    return args


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, args, with_label=True):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args
        self.with_label = with_label

    def __getitem__(self, idx):
        input_seq = self.tokenizer(self.data[idx]['input'] + ' <SEP> ')
        idx_of_sep = len(input_seq['input_ids'])

        if self.with_label:
            seq = self.data[idx]['input'] + ' <SEP> ' + self.data[idx]['target'] + ' ' + self.tokenizer.eos_token
            seq = self.tokenizer(seq)
        else:
            # this space is important to remove for inference, need to double check tokenization :-)
            input_seq = self.tokenizer(self.data[idx]['input'] + ' <SEP>')
            seq = input_seq

        seq['prompt_ends_idx'] = idx_of_sep - 1
        seq['instance_idx'] = idx
        return seq

    def __len__(self):
        return len(self.data)

    def get_instance_id(self, idx):
        return self.data[idx]['instance_id']


def compute_loss(input_ids, attention_mask, prompt_end_idx, logits):
    '''
    returns prompt_loss and completion_loss
    '''
    loss_fn = torch.nn.CrossEntropyLoss()

    logits = logits[:, :-1]
    targets = input_ids[:, 1:]
    targets_mask = attention_mask[:, 1:]

    # make prompt/completion mask
    idxs = torch.arange(targets_mask.shape[1], device=attention_mask.device).repeat((targets_mask.shape[0], 1))
    is_prompt = (idxs < (prompt_end_idx[:, None]-1)) * 1
    is_completion = (idxs >= (prompt_end_idx[:, None]-1)) * targets_mask
    is_first_tok_completion = (idxs == (prompt_end_idx[:, None]-1)) * targets_mask

    targets_prompt = targets * is_prompt + -100 * (1-is_prompt)
    targets_completion = targets * is_completion + -100 * (1-is_completion)

    # this could probably be refactored to be more efficient
    loss_prompt = loss_fn(logits.transpose(2, 1), targets_prompt)
    loss_completion = loss_fn(logits.transpose(2, 1), targets_completion)

    # compute some stats
    with torch.no_grad():
        preds = logits.argmax(2)
        is_token_acc = (preds==targets)
        stats = {
            'acc_prompt': ((is_token_acc*is_prompt).sum(axis=1)*1.0 / (is_prompt.sum(axis=1))).mean(),
            'acc_completion': ((is_token_acc*is_completion).sum(axis=1)*1.0 / (is_completion.sum(axis=1))).mean(),
            'n_inst_with_correct_first_tok_completion': ((is_token_acc*is_first_tok_completion).sum(axis=1)*1.0).sum(), # good for multichoice
            'n_toks_prompt': (1.0*is_prompt).sum(axis=1).mean(),
            'n_toks_completion': (1.0*is_completion).sum(axis=1).mean()
        }

    return loss_prompt, loss_completion, stats


def clamp_llamarmsnorm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    info = torch.finfo(input_dtype)
    hidden_states = torch.clamp(hidden_states, min=info.min, max=info.max) # sometimes there are infs which causes nans...
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    hidden_states = torch.clamp(hidden_states, min=info.min, max=info.max) # sometimes there are infs which causes nans...
    return self.weight * hidden_states.to(input_dtype)


def main():
    args = parse_args()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    if 'GPTQ' in args.model:
        transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = clamp_llamarmsnorm_forward # hacky, but we need the clamps for now for GPTQ

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accelerator = accelerate.Accelerator()
    mainproc = accelerator.is_local_main_process
    if not args.skip_save and mainproc:
        print('saving to {}'.format(args.output_path))

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    with open(args.train) as f:
        train = [json.loads(line) for line in f.readlines()]
    with open(args.val) as f:
        val = [json.loads(line) for line in f.readlines()]

    train_loader, val_loader = map(
        lambda x: SequenceDataset(x, tokenizer, args),
        [train, val]
    )

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    data_collator = transformers.DataCollatorWithPadding(
        tokenizer,
        return_tensors='pt'
    )

    train_loader = torch.utils.data.DataLoader(
        train_loader, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size, num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_loader, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size, num_workers=4
    )

    if 'GPTQ' in args.model:
        # do we need to disable exllama for training? maybe...
        quantization_config = transformers.GPTQConfig(bits=4, disable_exllama=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if args.use_bfloat else torch.float16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    else:
        if args.load_in_4bit:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16 if args.use_bfloat else torch.float16,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.bfloat16 if args.use_bfloat else torch.float16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type='nf4'))
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        else:
            if args.use_bfloat:
                model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
            else:
                model = transformers.AutoModelForCausalLM.from_pretrained(args.model)

    print('The model\'s dtype is {}'.format(model.dtype))

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=16, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        if mainproc:
            model.print_trainable_parameters()

    trainable_params = model.parameters()
    if not args.use_adafactor:
        optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    else:
        optim = transformers.Adafactor(trainable_params, lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)

    best_val_acc, best_val_loss, not_improved_epoch = 0.0, np.inf, 0

    if mainproc:
        if args.use_lora:
            tmpfile = tempfile.TemporaryDirectory()
        else:
            tmpfile = tempfile.NamedTemporaryFile()
        print('using tempfile {}'.format(tmpfile.name))

    model, optim, train_loader, val_loader = accelerator.prepare(model, optim, train_loader, val_loader)
    streamer = transformers.TextStreamer(tokenizer)

    for epoch in range(args.n_epochs):
        if mainproc:
            print('Epoch {}'.format(epoch))
        for mode in ['train', 'val']:
            if mode == 'train':
                if args.just_val: continue
                model.train()
                bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), disable=not mainproc)
            else:
                model.eval()
                bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), disable=not mainproc)

            epoch_stats = {
                'n_batch': 0.0, 'n_exs': 0.0, 'running_sum_loss': 0.0,
                'running_sum_prompt_loss': 0.0, 'running_sum_completion_loss': 0.0,
                'running_sum_prompt_acc': 0.0, 'running_sum_completion_acc': 0.0,
                'running_sum_first_tok_completion_acc': 0.0
            }

            for i, batch in bar:
                with torch.set_grad_enabled(mode=='train'):
                    output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    prompt_loss, completion_loss, stats = compute_loss(batch['input_ids'], batch['attention_mask'], batch['prompt_ends_idx'], output['logits'])
                    loss = args.prompt_loss_weight * prompt_loss.mean() + completion_loss.mean()
                    if mode == 'train':
                        loss_scaled = loss / args.gradient_accumulation_steps
                        accelerator.backward(loss_scaled)
                        if i % args.gradient_accumulation_steps == 0 or i == len(train_loader) - 1:
                            optim.step()
                            optim.zero_grad()
                    if mainproc:
                        epoch_stats['n_batch'] += 1
                        epoch_stats['n_exs'] += output['logits'].shape[0]
                        epoch_stats['running_sum_loss'] += loss.cpu().detach().numpy()
                        epoch_stats['running_sum_prompt_loss'] += prompt_loss.cpu().detach().numpy()
                        epoch_stats['running_sum_completion_loss'] += completion_loss.cpu().detach().numpy()
                        epoch_stats['running_sum_prompt_acc'] += stats['acc_prompt'].cpu().numpy()
                        epoch_stats['running_sum_completion_acc'] += stats['acc_completion'].cpu().numpy()
                        epoch_stats['running_sum_first_tok_completion_acc'] += stats['n_inst_with_correct_first_tok_completion'].cpu().numpy()

                        bar.set_description('loss = {:.6f} (loss prompt/completion = {:.3f}/{:.3f}; token acc prompt/completion/clf first tok = {:.2f}%/{:.2f}%/{:.2f}%)'.format(
                            epoch_stats['running_sum_loss'] / epoch_stats['n_batch'],
                            epoch_stats['running_sum_prompt_loss'] / epoch_stats['n_batch'],
                            epoch_stats['running_sum_completion_loss'] / epoch_stats['n_batch'],
                            100*epoch_stats['running_sum_prompt_acc'] / epoch_stats['n_batch'],
                            100*epoch_stats['running_sum_completion_acc'] / epoch_stats['n_batch'],
                            100*epoch_stats['running_sum_first_tok_completion_acc'] / epoch_stats['n_exs']))

                        if mode == 'val' and args.generate_during_val:
                            print(tokenizer.decode(batch['input_ids'][0][:batch['prompt_ends_idx'][0]]))
                            sample = accelerator.unwrap_model(model).generate(
                                input_ids=batch['input_ids'][:1][:, :batch['prompt_ends_idx'][0]],
                                max_new_tokens=128, streamer=streamer
                            )
                            print('prediction: {}'.format(tokenizer.decode(sample[0][batch['prompt_ends_idx'][0]:]).strip()))
                            print('~'*10)

            if mode == 'val' and mainproc:
                val_loss = epoch_stats['running_sum_loss'] / epoch_stats['n_batch']
                print('we computed accuracy/loss over {} validation examples.'.format(epoch_stats['n_exs']))
                best_yet = val_loss < best_val_loss

                if best_yet:
                    print('{} is a better than than {}, saving weights!'.format(
                        val_loss,
                        best_val_loss))
                    best_val_loss = val_loss
                    if not args.skip_save:
                        if not args.use_lora:
                            torch.save(
                                {'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                                 'args': vars(args)},
                                tmpfile.name)
                        else:
                            accelerator.unwrap_model(model).save_pretrained(tmpfile.name)
                        not_improved_epoch = 0
                else:
                    not_improved_epoch += 1

            if args.just_val: break

    accelerator.wait_for_everyone()
    if mainproc and not args.skip_save:
        args.output_path = args.output_path.format(best_val_loss)
        subprocess.call('cp -R {} {}'.format(tmpfile.name, args.output_path), shell=True)


if __name__ == '__main__':
    main()
