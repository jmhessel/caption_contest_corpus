'''
Trains a CLIP model for the multiple choice tasks

for task in {matching,ranking}; do for sp in {1,2,3,4}; do for lr in {.00001,.00005,.000005}; do accelerate launch train_clip.py $sp $task --warmup 200 --clip_model ViT-L/14@336px --pad 1 --lr $lr --use_accelerate 1 --batch_size 16 --n_epochs 12; done; done; done;
'''
import argparse
import numpy as np
import torch
import json
import pprint
import PIL
from PIL import Image, ImageDraw
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomGrayscale, ColorJitter
import tempfile
import tqdm
import os
import collections
import clip
import torchvision.transforms.functional as F
import accelerate
import random
import subprocess
import pprint
from datasets import load_dataset


class SquarePad:
    # https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, data, args, training=False):
        self.args = args
        self.data = data

        self.training = training
        if self.args.pad:
            self.preprocess = self._transform_train(args.input_resolution) if self.training else self._transform_test(args.input_resolution)
        else:
            self.preprocess = self._transform_train_pad(args.input_resolution) if self.training else self._transform_test_pad(args.input_resolution)

    def _transform_train_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_train(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomCrop(n_px),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def image_to_torch_tensor(self, image):
        image = self.preprocess(image)
        return image

    def __getitem__(self, idx):
        c_data = self.data[idx]
        if 'filepath' in c_data:
            image = Image.open(c_data['filepath'])
        else:
            image = c_data['image']
        choices = clip.tokenize(c_data['choices'], truncate=True)
        image = self.image_to_torch_tensor(image)
        to_ret = {'image':image, 'choices': choices, 'label': c_data['label']}
        if 'instance_id' in c_data:
            to_ret['instance_id'] = c_data['instance_id']
        return to_ret

    def __len__(self):
        return len(self.data)


def clip_forward(model, image, text):

    if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.parallel.DataParallel):
        image_features = model.module.encode_image(image)
        text_features = model.module.encode_text(text)
    else:
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return image_features, text_features


def add_prefix(instances, args):
    for inst in instances:
        inst['choices'] = [(args.prefix + ch).strip() for ch in inst['choices']]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('split', type=int, help='0,1,2,3,4 are cross-val splits 5 is leaderboard')
    parser.add_argument('task', type=str, choices=['matching', 'ranking'])

    parser.add_argument('--clip_model',
                        default='ViT-L/14@336px',
                        choices=['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16', 'RN50x16', 'RN50x64', 'ViT-L/14@336px', 'ViT-L/14'])

    parser.add_argument('--batch_size',
                        default=32,
                        type=int)

    parser.add_argument('--n_epochs',
                        type=int,
                        default=10)

    parser.add_argument('--lr',
                        type=float,
                        default=.00001)

    parser.add_argument('--use_accelerate',
                        type=int,
                        default=0,
                        help='if this flag is set, we will use huggingface accelerate intsead of dataparallel')

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help='how many steps for gradient accumulation')

    parser.add_argument('--pad',
                        type=int,
                        default=1,
                        help='if 0 we will do standard center crop, if 1 we will do pad.')

    parser.add_argument('--warmup',
                        type=int,
                        default=200,
                        help='how many steps of warmup should we use?')

    parser.add_argument('--force_run',
                        type=int,
                        default=0,
                        help='if 1, we will force the run even if the output already exists.')

    parser.add_argument('--prefix',
                        type=str,
                        default=None,
                        help='if this prefix is set, it will be appended to the input.')

    args = parser.parse_args()

    if args.prefix and ('+' in args.prefix or '~' in args.prefix):
        print('We dont support plus signs or tildes in prefixes.')
        quit()

    args.output_path = ('task={}'.format(args.task) +
                        '~split={}'.format(args.split) +
                        '~val{}'.format('acc') + '={:.5f}' + '~pad={}'.format(args.pad) +
                        '~model=' + '{}'.format(args.clip_model.replace('/', '*')) +
                        ('' if not args.prefix else 'prefix=' + '+'.join(args.prefix.strip().split())) +
                        '~lr={}.pt'.format(args.lr))

    if not args.prefix:
        args.prefix = ''
    else:
        args.prefix = args.prefix.strip() + ' '

    if not args.force_run:
        toks = args.output_path.split('/')
        outdir = '/'.join(toks[:-1]) if len(toks) > 1 else '.'
        def fnameparse(x):
            return (x.split('~val')[0], '~'.join(x.split('~val')[1].split('~')[1:]))
        if fnameparse(args.output_path) in set([fnameparse(x) for x in os.listdir(outdir) if '.pt' in x]):
            print('{} done already, run with --force_run to run.'.format(args.output_path))
            quit()
    return args


def batch_to_device(batch, mode, args):
    image, choices, labels = batch['image'], batch['choices'], batch['label']
    if not args.use_accelerate or mode == 'val':
        image, choices, labels = map(
            lambda x: x.to(args.device),
            [image, choices, labels])

    return dict(zip(['image', 'choices', 'labels'],
                    [image, choices, labels]))


def convert_matching(inst, args, leaderboard_mode=False):
    '''
    standardizes into matching format
    '''
    new_inst = {}
    if leaderboard_mode:
        new_inst['choices'] = [inst['choices'][l] for l in 'ABCDE']
        new_inst['label'] = 0 # dummy
        new_inst['contest_number'] = 0 # dummy
        new_inst['instance_id'] = inst['instance_id']
    else:
        new_inst['choices'] = inst['caption_choices']
        new_inst['label'] = 'ABCDE'.index(inst['label'])
        new_inst['contest_number'] = inst['contest_number']


    if isinstance(inst['image'], str):
        new_inst['filepath'] = inst['image']
    elif isinstance(inst['image'], PIL.JpegImagePlugin.JpegImageFile):
        new_inst['image'] = inst['image']
    else:
        new_inst['filepath'] = inst['image']['path']

    return new_inst


def convert_quality(inst, args, leaderboard_mode=False):
    '''
    standardizes into ranking format

    if leaderboard mode, assume the input inst is from the leaderboard format.
    '''
    new_inst = {}
    if leaderboard_mode:
        new_inst['choices'] = [inst['choices'][l] for l in 'AB']
        new_inst['label'] = 0 # dummy
        new_inst['contest_number'] = 0 # dummy
        new_inst['instance_id'] = inst['instance_id']
    else:
        new_inst['choices'] = inst['caption_choices']
        new_inst['label'] = 'AB'.index(inst['label'])
        new_inst['contest_number'] = inst['contest_number']


    if isinstance(inst['image'], str):
        new_inst['filepath'] = inst['image']
    elif isinstance(inst['image'], PIL.JpegImagePlugin.JpegImageFile):
        new_inst['image'] = inst['image']
    else:
        new_inst['filepath'] = inst['image']['path']

    return new_inst


def main():
    args = parse_args()
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accelerator = accelerate.Accelerator()
    mainproc = accelerator.is_local_main_process

    model, preprocess = clip.load(args.clip_model, jit=False, device='cpu')
    model.float()
    try:
        args.input_resolution = model.visual.input_resolution
    except:
        args.input_resolution = model.input_resolution


    if args.task == 'matching':
        split_name = 'matching_from_pixels' if args.split in [0, 5] else 'matching_from_pixels_{}'.format(args.split)
    elif args.task == 'ranking':
        split_name = 'ranking_from_pixels' if args.split in [0, 5] else 'ranking_from_pixels_{}'.format(args.split)

    data = load_dataset("jmhessel/newyorker_caption_contest", split_name)

    train, val, test = data['train'], data['validation'], data['test']
    if args.split == 5:
        train = list(train) + list(val)
        val = test
        test = []

    print('train/val/test datapoints: {}/{}/{}'.format(*map(len, [train, val, test])))

    if args.task == 'matching':
        train = [convert_matching(t, args) for t in train]
        val = [convert_matching(t, args) for t in val]
    elif args.task == 'ranking':
        train = [convert_quality(t, args) for t in train]
        val = [convert_quality(t, args) for t in val]
    else:
        raise NotImplementedError

    add_prefix(train, args)
    add_prefix(val, args)

    train_loader = CLIPDataset(train, args, training=True)
    val_loader = CLIPDataset(val, args, training=False)

    train_loader = torch.utils.data.DataLoader(
        train_loader, shuffle=True, batch_size=args.batch_size, num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        val_loader, shuffle=False, batch_size=args.batch_size, num_workers=4
    )

    if not args.use_accelerate and torch.cuda.device_count() > 1:
        print('Lets use', torch.cuda.device_count(), 'GPUs!')
        model = torch.nn.DataParallel(model)

    if not args.use_accelerate:
        model.to(args.device)
    else:
        args.device = accelerator.device

    trainable_params = model.parameters()
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)

    def lr_lambda(current_step):
        if current_step < args.warmup:
            mul = float(current_step) / float(max(1.0, args.warmup))
        else:
            mul = 1.0
        return mul

    schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    best_val_acc, best_val_loss, not_improved_epoch = 0.0, np.inf, 0

    if mainproc:
        tmpfile = tempfile.NamedTemporaryFile()
        print('using tempfile {}'.format(tmpfile.name))

    if args.use_accelerate:
        model, optim, train_loader, schedule = accelerator.prepare(model, optim, train_loader, schedule)
    try:
        logit_scale = model.module.logit_scale
    except:
        logit_scale = model.logit_scale

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epochs):
        if mainproc:
            print('Epoch {}'.format(epoch))
        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
                bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), disable=not mainproc)
            else:
                model.eval()
                bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader), disable=not mainproc)
            n, running_sum_loss, n_exs, running_sum_accs = 0, 0, 0, 0
            all_preds, all_labels = [], []

            for i, batch in bar:
                with torch.set_grad_enabled(mode=='train'):
                    batch = batch_to_device(batch, mode, args)
                    n_choice = batch['choices'].shape[1]
                    batch['choices'] = batch['choices'].reshape((-1, 77))
                    image_features, text_features = clip_forward(model, batch['image'], batch['choices'])
                    text_features = text_features.reshape((image_features.shape[0], n_choice, -1))
                    image_features = torch.unsqueeze(image_features, 1)
                    logits = logit_scale.exp() * (image_features * text_features).sum(2)
                    preds = logits.argmax(1)
                    all_preds.extend(preds.detach().cpu().numpy().tolist())
                    all_labels.extend(batch['labels'].detach().cpu().numpy().tolist())
                    loss = loss_fn(logits, batch['labels'])

                    if mode == 'train':
                        loss_scaled = loss / args.gradient_accumulation_steps
                        if args.use_accelerate:
                            accelerator.backward(loss_scaled)
                        else:
                            loss_scaled.backward()

                        if i % args.gradient_accumulation_steps == 0 or i == len(train_loader) - 1:
                            optim.step()
                            optim.zero_grad()
                            schedule.step()

                    n += 1
                    n_exs += image_features.shape[0]
                    running_sum_loss += loss.cpu().detach().numpy()
                    running_sum_accs += np.sum(preds.detach().cpu().numpy() == batch['labels'].detach().cpu().numpy())
                    bar.set_description('loss = {:.6f} acc = {:.6f}'.format(running_sum_loss / n, running_sum_accs / n_exs))


            if mode == 'val' and mainproc:
                val_acc = running_sum_accs / n_exs
                val_loss = running_sum_loss / n
                print('we computed accuracy/loss over {} validation examples.'.format(n_exs))

                best_yet = val_acc > best_val_acc

                if best_yet:
                    print('{} is a better than than {}, saving weights!'.format(
                        val_acc,
                        best_val_acc))

                    best_val_acc = val_acc

                    if args.use_accelerate:
                        torch.save(
                            {'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                             'args': vars(args)},
                            tmpfile.name)
                    else:
                        try:
                            torch.save(
                                {'model_state_dict': model.module.state_dict(),
                                 'optimizer_state_dict': optim.state_dict(),
                                 'args': vars(args)},
                                tmpfile.name)
                        except:
                            torch.save(
                                {'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optim.state_dict(),
                                 'args': vars(args)},
                                tmpfile.name)
                        not_improved_epoch = 0
                else:
                    not_improved_epoch += 1

    accelerator.wait_for_everyone()
    if mainproc:
        args.output_path = args.output_path.format(best_val_acc)
        subprocess.call('cp {} {}'.format(tmpfile.name, args.output_path), shell=True)


if __name__ == '__main__':
    main()
