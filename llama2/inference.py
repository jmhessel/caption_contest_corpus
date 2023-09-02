'''
Inference, only single GPU supported for now.
'''
import argparse
import collections
import tqdm
import json
import numpy as np
import os
import transformers
import tempfile
import torch
import random
import subprocess
import pprint
import train
from peft import PeftModel


def load_args_from_fname(args):
    if args.checkpoint.endswith('/'):
        fname_to_parse = args.checkpoint[:-1].split('/')[-1].split('.pt')[0]
    else:
        fname_to_parse = args.checkpoint.split('/')[-1].split('.pt')[0]
    toks = fname_to_parse.split('~')
    for t in toks:
        if '=' not in t:
            args.task_name = t
        else:
            k, v = t.split('=')
            if k == 'model':
                args.model = v.replace('+', '/')
            elif k == 'lr':
                args.lr = float(v)
            elif k == '4bit':
                args.load_in_4bit = int(v)
            elif k == 'promptloss':
                args.prompt_loss_weight = float(v)
            elif k == 'lora':
                args.use_lora = int(v)


def batch_to_device(batch, args):
    res = {}
    for k, v in batch.items():
        res[k] = v.to(args.device)
    return res


def make_output_fname(args):
    output_f = str(args.checkpoint)
    if output_f.endswith('.pt'): output_f = output_f[:-3]
    if output_f.endswith('/'): output_f = output_f[:-1]
    output_f = output_f.split('/')[-1]
    output_f += '~preds_for_instances={}'.format(args.instances.split('/')[-1].replace('.jsonl', ''))
    output_f += '.json'
    args.output_f = output_f
    print('saving predictions to {}'.format(args.output_f))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('instances')

    parser.add_argument('--tokenizer',
                        default=None,
                        help='if the tokenizer and the model are saved separately, then use this to specify just the tokenizer',
                        type=str)

    parser.add_argument('--batch_size',
                        default=32,
                        type=int)

    parser.add_argument('--use_bfloat',
                        default=0,
                        help='should we use bfloat?',
                        type=int)

    parser.add_argument('--temp',
                        default=.8,
                        type=float)

    parser.add_argument('--top_p',
                        default=.95,
                        type=float)

    parser.add_argument('--prompt_delimiter_string',
                        type=str,
                        default='<SEP>',
                        help='what string should delimit the prompt/completion?')

    args = parser.parse_args()

    load_args_from_fname(args)
    make_output_fname(args)

    if not args.tokenizer:
        args.tokenizer = args.model

    return args


def main():
    args = parse_args()
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    if 'GPTQ' in args.model:
        transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = train.clamp_llamarmsnorm_forward # hacky, but we need the clamps

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    generation_tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, padding_side='left')
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    with open(args.instances) as f:
        val = [json.loads(line) for line in f.readlines()]

    val_loader_with_label = train.SequenceDataset(val, tokenizer, args)
    val_loader_without_label = train.SequenceDataset(val, generation_tokenizer, args, with_label=False)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if not generation_tokenizer.pad_token_id:
        generation_tokenizer.pad_token_id = generation_tokenizer.unk_token_id

    data_collator = transformers.DataCollatorWithPadding(
        tokenizer,
        return_tensors='pt'
    )
    data_collator_for_gen = transformers.DataCollatorWithPadding(
        generation_tokenizer,
        return_tensors='pt'
    )

    val_loader_with_label = torch.utils.data.DataLoader(
        val_loader_with_label, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size, num_workers=4
    )

    val_loader_without_label = torch.utils.data.DataLoader(
        val_loader_without_label, shuffle=False, collate_fn=data_collator_for_gen, batch_size=args.batch_size, num_workers=4
    )

    if 'GPTQ' in args.model:
        quantization_config = transformers.GPTQConfig(bits=4, disable_exllama=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if args.use_bfloat else torch.float16)
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
            print('loaded quantized model')
        else:
            if args.use_bfloat:
                model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
            else:
                model = transformers.AutoModelForCausalLM.from_pretrained(args.model)

        if not args.use_lora:
            state = torch.load(args.checkpoint)
            state['model_state_dict'] = {k.replace('module.', '') : v for k, v in state['model_state_dict'].items()}
            model.load_state_dict(state['model_state_dict'])
        else:
            model = PeftModel.from_pretrained(model, args.checkpoint)


    print('The model\'s dtype is {}'.format(model.dtype))

    best_val_acc, best_val_loss, not_improved_epoch = 0.0, np.inf, 0

    model.to(args.device)
    model.eval()

    bar = tqdm.tqdm(enumerate(zip(val_loader_with_label, val_loader_without_label)), total=len(val_loader_with_label))

    epoch_stats = {
        'n_batch': 0.0, 'n_exs': 0.0, 'running_sum_loss': 0.0,
        'running_sum_prompt_loss': 0.0, 'running_sum_completion_loss': 0.0,
        'running_sum_prompt_acc': 0.0, 'running_sum_completion_acc': 0.0,
        'running_sum_first_tok_completion_acc': 0.0
    }

    instanceid2pred = {}
    for i, (with_label, without_label) in bar:
        with torch.no_grad():

            with_label = batch_to_device(with_label, args)
            without_label = batch_to_device(without_label, args)
            ids = [val_loader_with_label.dataset.get_instance_id(idx) for idx in  with_label['instance_idx'].cpu().numpy()]
            output = model(input_ids=with_label['input_ids'], attention_mask=with_label['attention_mask'])

            prompt_loss, completion_loss, stats = train.compute_loss(with_label['input_ids'], with_label['attention_mask'], with_label['prompt_ends_idx'], output['logits'])
            loss = args.prompt_loss_weight * prompt_loss.mean() + completion_loss.mean()

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

            if args.temp == 0:
                sample = model.generate(
                    input_ids=without_label['input_ids'],
                    attention_mask=without_label['attention_mask'],
                    max_new_tokens=1024,
                    do_sample=False
                )
            else:
                sample = model.generate(
                    input_ids=without_label['input_ids'],
                    attention_mask=without_label['attention_mask'],
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=args.temp,
                    top_p=args.top_p)

            sample = [tokenizer.decode(s, skip_special_tokens=True).split(args.prompt_delimiter_string)[-1].strip() for s in sample]

            for cid, s in zip(ids, sample):
                instanceid2pred[cid] = s

    print('saving {} predictions to {}'.format(len(instanceid2pred), args.output_f))
    with open(args.output_f, 'w') as f:
        f.write(json.dumps(instanceid2pred))


if __name__ == '__main__':
    main()
