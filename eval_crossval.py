'''
Official cross-val evaluation script.

Give this script the predictions in the form of {'instance_id': 'prediction'} for any of the tasks, and this will compute the appropriate metrics.

To run with a set of dummy predictions, you can run:

for task in {matching,ranking,explanation}; do python eval_crossval.py None --just_test_eval 1 --task $task; done;
'''
import argparse
import json
import numpy as np
import pprint
from datasets import load_dataset
from evaluate import load


_bleu, _rouge = None, None

def get_generation_metrics():
    global _bleu, _rouge
    if _bleu is None:
        _bleu = load("sacrebleu")
    if _rouge is None:
        _rouge = load("rouge")
    return _bleu, _rouge


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'predictions',
        type=str,
        nargs='+',
        help='jsons mapping from {"instance_id": "prediction"}')

    parser.add_argument(
        '--task',
        default='matching',
        type=str,
        choices=['matching', 'ranking', 'explanation'])

    parser.add_argument(
        '--just_test_eval',
        default=0,
        type=int,
        help='if 1, then we will just do a valid test evaluation run',
        choices=[0,1])
    return parser.parse_args()


def create_dummy_predictions(query_ids, task):
    query_ids = list(query_ids)
    if task == 'matching':
        return {query_ids[idx]: 'ABCDE'[r_idx] for idx, r_idx in enumerate(np.random.choice(5, size=len(query_ids)))}
    elif task == 'ranking':
        return {query_ids[idx]: 'AB'[r_idx] for idx, r_idx in enumerate(np.random.choice(2, size=len(query_ids)))}
    elif task == 'explanation':
        return {query_ids[idx]: 'The joke is funny because the caption matches.' for idx in range(len(query_ids))}


def main():
    np.random.seed(1)
    args = parse_args()

    predictions = {}
    if not args.just_test_eval:
        for p in args.predictions:
            with open(p) as f:
                predictions = {**predictions, **json.load(f)}

    split2metrics = {}
    for split in range(5):
        split_name = '{}_{}'.format(args.task, split) if split != 0 else '{}'.format(args.task)
        dataset = list(load_dataset("jmhessel/newyorker_caption_contest", split_name, split='test'))
        query_ids = set([d['instance_id'] for d in dataset])
        if args.just_test_eval:
            predictions = create_dummy_predictions(query_ids, args.task)

        if not query_ids.issubset(set(predictions.keys())):
            print('Cannot evaluate {} split {}, missing {}/{} predictions; skipping.'.format(
                args.task, split, len(query_ids-set(predictions.keys())), len(query_ids)))
            continue

        if args.task == 'matching':
            id2label = {d['instance_id']: d['label'] for d in dataset}
            split2metrics[split] = {
                'accuracy': np.mean([id2label[q] == predictions[q] for q in query_ids]),
                'n': len(query_ids)}

        elif args.task == 'ranking':
            acc_ny, acc_crowd = [], []
            for d in dataset:
                corr = float(d['label'] == predictions[d['instance_id']])
                if d['winner_source'] == 'crowd_winner':
                    acc_crowd.append(corr)
                else:
                    acc_ny.append(corr)

            split2metrics[split] = {
                'accuracy_overall': np.mean(acc_ny + acc_crowd),
                'n_overall': len(acc_ny + acc_crowd),
                'accuracy_ny': np.mean(acc_ny),
                'n_ny': len(acc_ny),
                'accuracy_crowd': np.mean(acc_crowd),
                'n_crowd': len(acc_crowd)
            }

        elif args.task == 'explanation':
            print('Warning: these simple automatic evaluation metrics may not correlate with human judgement for explanation generation.')
            hypotheses = [predictions[q] for q in query_ids]
            id2label = {d['instance_id']: d['label'] for d in dataset}
            references = [id2label[q] for q in query_ids]

            n_null = 0
            for h in hypotheses:
                if not h:
                    n_null += 1
            if n_null > 0:
                print('{} null predictions, skipping split.'.format(n_null))
                continue

            split2metrics[split] = {}
            bleu, rouge = get_generation_metrics()
            for name, m in zip(['bleu', 'rouge'], [bleu, rouge]):
                if m is None: continue
                kwargs = {}
                test_res = m.compute(predictions=hypotheses, references=references, **kwargs)
                if name == 'bleu':
                    split2metrics[split][name] = test_res['score']
                if name == 'rouge':
                    split2metrics[split][name] = np.mean(test_res['rougeL'])

                split2metrics[split]['n'] = len(hypotheses)

    if len(split2metrics) == 5: # full cross-val results!

        print('Full averaged cross-validation results for task={}'.format(args.task))
        dircomp = '***** these are directly comparable to the reported metrics in the paper! https://arxiv.org/abs/2209.06293 (Table 2 or Table 5)*****'
        print(dircomp)
        metrics = split2metrics[0].keys()
        averaged_scores = {}
        for m in metrics:
            averaged_scores[m] = np.mean([v[m] for v in split2metrics.values()])
        pprint.pprint(averaged_scores)
        print('*'*len(dircomp))

    print('per split metrics:')
    pprint.pprint(split2metrics)


if __name__ == '__main__':
    main()
