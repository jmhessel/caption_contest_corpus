'''
Get leaderboard predictions for a given CLIP model
'''
import argparse
import numpy as np
import torch
import json
import pprint
import tqdm
import os
import collections
import clip
import accelerate
import random
import subprocess
import pprint
import train_clip as trainlib
from datasets import load_dataset


def get_args_from_checkpoint_filename(fname):
    kvars = fname.split('/')[-1].split('.pt')[0]
    kvars = kvars.split('~')
    kvars = {x.split('=')[0] : x.split('=')[1] for x in kvars}
    if 'model' in kvars:
        kvars['model'] = kvars['model'].replace('*', '/')

    for k in ['pad', 'split']:
        if k in kvars:
            kvars[k] = int(kvars[k])

    return kvars['task'], kvars['split'], kvars['pad'], kvars['model']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('clip_model_path',
                        help='either a path to model, or "zero_shot" for zero shot evaluation')

    parser.add_argument('leaderboard_dir',
                        help='contains instances and images')

    parser.add_argument('--batch_size',
                        default=32,
                        type=int)

    parser.add_argument('--prefix',
                        default=None,
                        type=str,
                        help='if this prefix is set, it will be appended to the input.')

    ### these arguments will be autospecified if clip_model_path is specified, else, for zero shot, you need to specify these.
    parser.add_argument('--task',
                        default=None,
                        choices=['matching', 'ranking'],
                        type=str,
                        help='what task are we looking at?')

    parser.add_argument('--pad',
                        default=1,
                        type=int,
                        help='if 0 we will do standard center crop, if 1 we will do pad.')

    parser.add_argument('--clip_model',
                        default='ViT-L/14@336px',
                        type=str,
                        choices=['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16', 'RN50x16', 'RN50x64', 'ViT-L/14@336px', 'ViT-L/14'])

    ###

    args = parser.parse_args()

    if args.clip_model_path != 'zero_shot':
        args.task, args.split, args.pad, args.clip_model = get_args_from_checkpoint_filename(args.clip_model_path)

    if args.clip_model_path != 'zero_shot':
        args.zero_shot_mode = False
        args.prefix = '' # assume fine-tuned models are trained with no prefix.
        args.output = args.clip_model_path.replace('.pt', '~results.json')
    else:
        args.zero_shot_mode = True
        if not args.prefix:
            args.prefix = ''
        else:
            args.prefix = args.prefix.strip()
        args.output = 'task={}~split={}~prefix={}~pad={}~model={}~results.json'.format(
            args.task,
            5,
            '+'.join(args.prefix.strip().split()),
            args.pad,
            args.clip_model.replace('/', '*'))
        args.prefix = args.prefix + ' '

    print('padding={}, backbone={}'.format(args.pad, args.clip_model))
    args.use_accelerate = False

    print('writing all results to {}'.format(args.output))

    return args


def load_leaderboard_instances(leaderboard_dir):
    with open(leaderboard_dir + '/instances.json') as f:
        instances = json.load(f)

    for inst in instances:
        inst['image'] = leaderboard_dir + '/' + inst['image']

    return instances

def main():
    args = parse_args()
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    test = load_leaderboard_instances(args.leaderboard_dir)
    if args.task == 'matching':
        test = [trainlib.convert_matching(t, args, leaderboard_mode=True) for t in test]
    elif args.task == 'ranking':
        test = [trainlib.convert_quality(t, args, leaderboard_mode=True) for t in test]
    else:
        raise NotImplementedError

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(args.clip_model, jit=False)
    model.float()


    # load model #
    if 'zero_shot' not in args.clip_model_path:
        print('Getting model weights from {}'.format(args.clip_model_path))
        state = torch.load(args.clip_model_path)
        state['model_state_dict'] = {k.replace('module.', '') : v for k, v in state['model_state_dict'].items()}
        model.load_state_dict(state['model_state_dict'])
    else:
        print('doing zero shot with {}!'.format(args.clip_model))

    model.eval()
    try:
        args.input_resolution = model.visual.input_resolution
    except:
        args.input_resolution = model.input_resolution

    trainlib.add_prefix(test, args)


    test_loader = torch.utils.data.DataLoader(
        trainlib.CLIPDataset(test, args, training=False),
        shuffle=False, batch_size=args.batch_size, num_workers=4)

    try:
        logit_scale = model.module.logit_scale
    except:
        logit_scale = model.logit_scale

    loss_fn = torch.nn.CrossEntropyLoss()

    bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))

    all_preds, all_instances = [], []

    for i, batch in bar:
        with torch.no_grad():
            instances = batch['instance_id']
            del batch['instance_id']
            batch = trainlib.batch_to_device(batch, 'val', args)
            n_choice = batch['choices'].shape[1]
            batch['choices'] = batch['choices'].reshape((-1, 77))
            image_features, text_features = trainlib.clip_forward(model, batch['image'], batch['choices'])
            text_features = text_features.reshape((image_features.shape[0], n_choice, -1))
            image_features = torch.unsqueeze(image_features, 1)
            logits = logit_scale.exp() * (image_features * text_features).sum(2)
            preds = logits.argmax(1).cpu().numpy().tolist()
            all_preds.extend(['ABCDE'[p] for p in preds])
            all_instances.extend(list(instances))

    output_dict = dict(zip(all_instances, all_preds))

    with open(args.output, 'w') as f:
        f.write(json.dumps(output_dict))

if __name__ == '__main__':
    main()
