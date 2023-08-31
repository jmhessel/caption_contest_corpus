'''
Load some toy tasks
'''
import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'task',
        help='which task?',
        choices=['joke_explanation', 'joke_matching', 'joke_ranking'],
    )
    
    return parser.parse_args()


def get_joke_explanation_instances():
    from datasets import load_dataset

    dset = load_dataset("jmhessel/newyorker_caption_contest", "explanation")

    res = {}
    for spl, spl_name in zip([dset['train'], dset['validation'], dset['test']],
                             ['train', 'val', 'test']):
        cur_spl = []
        for inst in list(spl):
            inp = inst['from_description']
            targ = inst['label']
            cur_spl.append({'input': inp, 'target': targ, 'instance_id': inst['instance_id']})

        res[spl_name] = cur_spl

    return res['train'], res['val'], res['test']


def get_joke_matching_instances():
    from datasets import load_dataset

    dset = load_dataset("jmhessel/newyorker_caption_contest", "matching")

    res = {}
    for spl, spl_name in zip([dset['train'], dset['validation'], dset['test']],
                             ['train', 'val', 'test']):
        cur_spl = []
        for inst in list(spl):
            inp = inst['from_description']
            targ = inst['label']
            cur_spl.append({'input': inp, 'target': targ, 'instance_id': inst['instance_id']})
        res[spl_name] = cur_spl

    return res['train'], res['val'], res['test']


def get_joke_ranking_instances():
    from datasets import load_dataset

    dset = load_dataset("jmhessel/newyorker_caption_contest", "ranking")

    res = {}
    for spl, spl_name in zip([dset['train'], dset['validation'], dset['test']],
                             ['train', 'val', 'test']):
        cur_spl = []
        for inst in list(spl):
            inp = inst['from_description']
            targ = inst['label']
            cur_spl.append({'input': inp, 'target': targ, 'instance_id': inst['instance_id']})
        res[spl_name] = cur_spl

    return res['train'], res['val'], res['test']
    

def main():
    args = parse_args()

    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    
    data_fn, base_fn = None, None
    if args.task == 'joke_explanation':
        data_fn = get_joke_explanation_instances
        base_fn = '{}_joke_explain.jsonl'
    elif args.task == 'joke_matching':
        data_fn = get_joke_matching_instances
        base_fn = '{}_joke_matching.jsonl'
    elif args.task == 'joke_ranking':
        data_fn = get_joke_ranking_instances
        base_fn = '{}_joke_ranking.jsonl'
    else:
        raise NotImplementedError

    train, val, test = data_fn()
    print('train/val/test {}/{}/{}'.format(*map(len, [train, val, test])))
    for insts, name in zip([train, val, test], ['train', 'val', 'test']):
        with open('datasets/' + base_fn.format(name), 'w') as f:
            f.write('\n'.join([json.dumps(i) for i in insts]))
    

if __name__ == '__main__':
    main()
