'''
From description zero/few shot results for GPT 3.5/4
'''
import argparse
from datasets import load_dataset
import pprint
import openai
import numpy as np
import json
import os
import time
import collections
import prompts
import copy
import hashlib
import tqdm

# matching prompts
_PROMPT_SYSTEM_MATCHING_V1, _PROMPT_USER_MATCHING_V1, _RESPONSE_ASSISTANT_MATCHING_V1 = (
    prompts._PROMPT_SYSTEM_MATCHING_V1,
    prompts._PROMPT_USER_MATCHING_V1,
    prompts._RESPONSE_ASSISTANT_MATCHING_V1)

# ranking prompts
_PROMPT_SYSTEM_RANKING_V1, _PROMPT_USER_RANKING_V1, _RESPONSE_ASSISTANT_RANKING_V1 = (
    prompts._PROMPT_SYSTEM_RANKING_V1,
    prompts._PROMPT_USER_RANKING_V1,
    prompts._RESPONSE_ASSISTANT_RANKING_V1)

# explanation prompts
_PROMPT_SYSTEM_EXPLANATION_V1, _PROMPT_USER_EXPLANATION_V1, _RESPONSE_ASSISTANT_EXPLANATION_V1 = (
    prompts._PROMPT_SYSTEM_EXPLANATION_V1,
    prompts._PROMPT_USER_EXPLANATION_V1,
    prompts._RESPONSE_ASSISTANT_EXPLANATION_V1)

# matching prompts direct answer
_PROMPT_SYSTEM_MATCHING_DIRECT_ANSWER_V1, _PROMPT_USER_MATCHING_DIRECT_ANSWER_V1, _RESPONSE_ASSISTANT_MATCHING_DIRECT_ANSWER_V1 = (
    prompts._PROMPT_SYSTEM_MATCHING_DIRECT_ANSWER_V1,
    prompts._PROMPT_USER_MATCHING_DIRECT_ANSWER_V1,
    prompts._RESPONSE_ASSISTANT_MATCHING_DIRECT_ANSWER_V1)

# ranking prompts direct answer
_PROMPT_SYSTEM_RANKING_DIRECT_ANSWER_V1, _PROMPT_USER_RANKING_DIRECT_ANSWER_V1, _RESPONSE_ASSISTANT_RANKING_DIRECT_ANSWER_V1 = (
    prompts._PROMPT_SYSTEM_RANKING_DIRECT_ANSWER_V1,
    prompts._PROMPT_USER_RANKING_DIRECT_ANSWER_V1,
    prompts._RESPONSE_ASSISTANT_RANKING_DIRECT_ANSWER_V1)

# explanation prompts direct answer
_PROMPT_SYSTEM_EXPLANATION_DIRECT_ANSWER_V1, _PROMPT_USER_EXPLANATION_DIRECT_ANSWER_V1, _RESPONSE_ASSISTANT_EXPLANATION_DIRECT_ANSWER_V1 = (
    prompts._PROMPT_SYSTEM_EXPLANATION_DIRECT_ANSWER_V1,
    prompts._PROMPT_USER_EXPLANATION_DIRECT_ANSWER_V1,
    prompts._RESPONSE_ASSISTANT_EXPLANATION_DIRECT_ANSWER_V1)

# ICL examples for extraction
_PROMPT_SYSTEM_ANSWER_EXTRACTION_V1, _ICL_SYSTEM_ANSWER_EXTRACTION_V1 = (
    prompts._PROMPT_SYSTEM_ANSWER_EXTRACTION_V1,
    prompts._ICL_SYSTEM_ANSWER_EXTRACTION_V1)


_FD_MODE = 4

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--split',
        default=0,
        type=int)

    parser.add_argument(
        '--limit',
        default=-1,
        type=int)

    parser.add_argument(
        '--task',
        default='matching',
        choices=['matching', 'ranking', 'explanation', 'explanation_perplexity'])

    parser.add_argument(
        '--val',
        default=1,
        type=int,
        help='1 is validation set, 0 is test set')

    parser.add_argument(
        '--engine',
        default='gpt-3.5-turbo',
        type=str)

    parser.add_argument(
        '--shots',
        default=0,
        help='Set this to how many few shot shots you want to use.',
        type=int)

    parser.add_argument(
        '--fd_mode',
        default=4,
        help='see "format_chatgpt_input',
        type=int)

    args = parser.parse_args()

    if args.shots == 0:
        if args.fd_mode == 4:
            args.predictions_output_f = 'model={}~task={}~split={}.json'.format(
                args.engine, args.task, args.split)
        else:
            args.predictions_output_f = 'model={}~task={}~fdmode={}~split={}.json'.format(
                args.engine, args.task, args.fd_mode, args.split)

    else:
        if args.fd_mode == 4:
            args.predictions_output_f = 'model={}~shots={}~task={}~split={}.json'.format(
                args.engine, args.shots, args.task, args.split)
        else:
            args.predictions_output_f = 'model={}~shots={}~task={}~fdmode={}~split={}.json'.format(
                args.engine, args.shots, args.task, args.fd_mode, args.split)


    print('writing predictions to {}'.format(args.predictions_output_f))

    global _FD_MODE
    _FD_MODE = args.fd_mode

    return args


def format_chatgpt_input(d):
    '''
    mode=1 is location
    mode=2 is location/description
    mode=3 is location/description/uncanny
    mode=4 is location/description/uncanny/entities (from paper)
    '''

    global _FD_MODE
    mode = _FD_MODE

    # lets fix entities...
    fixed_entities = []
    for ent in d['entities']:
        ent = ent.split('#')[0]
        ent = ent.replace('&redirect=no', '')
        ent = ent.replace('https://en.wikipedia.org/?title=', '/wiki/')
        ent = ent.split('/wiki/')[1]
        ent = ent.replace('%27s', "'")
        ent = ent.replace('(disambiguation)', '')
        ent = ent.replace('_', ' ')
        ent = ' '.join(ent.split())
        fixed_entities.append(ent)

    if mode == 4:
        return 'scene location: {}\ndescription: {}\nuncanny description: {}\nentities: {}'.format(
            d['image_location'],
            d['image_description'],
            d['image_uncanny_description'],
        ', '.join(fixed_entities))
    if mode == 3:
        return 'scene location: {}\ndescription: {}\nuncanny description: {}'.format(
            d['image_location'],
            d['image_description'],
            d['image_uncanny_description'])
    if mode == 2:
        return 'scene location: {}\ndescription: {}'.format(
            d['image_location'],
            d['image_description'])
    if mode == 1:
        return 'scene location: {}'.format(
            d['image_location'])

    if mode == 0:
        return 'a new yorker cartoon'


def generate_request_matching_few_shot(query, few_shots):
    global _PROMPT_SYSTEM_MATCHING_DIRECT_ANSWER_V1, _PROMPT_USER_MATCHING_DIRECT_ANSWER_V1, _RESPONSE_ASSISTANT_MATCHING_DIRECT_ANSWER_V1

    messages = [{"role": "system", "content": _PROMPT_SYSTEM_MATCHING_DIRECT_ANSWER_V1}]

    for idx, fs in enumerate(few_shots):

        extra = 'Thanks! How about this one?\n' if idx != 0 else (_PROMPT_USER_MATCHING_DIRECT_ANSWER_V1 + '\n')
        messages.append({
            'role': 'user',
            'content': extra + '\n{}\nChoices:\n{}\n\nWhich of the 5 options (A, B, C, D, or E) is the caption that truly corresponds to the cartoon?'.format(
                format_chatgpt_input(fs),
                '\n'.join(['{}: {}'.format(cidx, c) for cidx, c in zip('ABCDE', fs['caption_choices'])]))})
        messages.append({
            'role': 'assistant',
            'content': 'Answer: {}'.format(fs['label'])})

    extra = 'Thanks! How about this one?\n'
    messages.append({
        'role': 'user',
        'content': extra + '\n{}\nChoices:\n{}\n\nWhich of the 5 options (A, B, C, D, or E) is the caption that truly corresponds to the cartoon?'.format(
            format_chatgpt_input(query),
            '\n'.join(['{}: {}'.format(cidx, c) for cidx, c in zip('ABCDE', query['caption_choices'])]))})

    return messages


def generate_request_matching(query, few_shots=None):
    if few_shots:
        return generate_request_matching_few_shot(query, few_shots)

    global _PROMPT_SYSTEM_MATCHING_V1, _PROMPT_USER_MATCHING_V1, _RESPONSE_ASSISTANT_MATCHING_V1

    messages = [{"role": "system", "content": _PROMPT_SYSTEM_MATCHING_V1}]

    messages.append({
        'role': 'user',
        'content': _PROMPT_USER_MATCHING_V1}
    )

    messages.append({
        'role': 'assistant',
        'content': _RESPONSE_ASSISTANT_MATCHING_V1}
    )

    messages.append({
        'role': 'user',
        'content': ('OK. Here is a description of the cartoon followed by the five choices.'
                    '\n\n{}\n\nChoices:\n{}\n\nWhich of the 5 options (A, B, C, D, or E) is the best fit? Think step-by-step and finish your response with "Answer: X" where X is either A, B, C, D, or E.'.format(
                        format_chatgpt_input(query),
                        '\n'.join(['{}: {}'.format(cidx, c) for cidx, c in zip('ABCDE', query['caption_choices'])])))})

    return messages


def generate_request_ranking_few_shot(query, few_shots):
    global _PROMPT_SYSTEM_RANKING_DIRECT_ANSWER_V1, _PROMPT_USER_RANKING_DIRECT_ANSWER_V1, _RESPONSE_ASSISTANT_RANKING_DIRECT_ANSWER_V1

    messages = [{"role": "system", "content": _PROMPT_SYSTEM_RANKING_DIRECT_ANSWER_V1}]

    for idx, fs in enumerate(few_shots):

        extra = 'Thanks! How about this one?\n' if idx != 0 else (_PROMPT_USER_RANKING_DIRECT_ANSWER_V1 + '\n')
        messages.append({
            'role': 'user',
            'content': extra + '\n{}\nChoices:\n{}\n\nWhich of the 2 options (A or B) is funnier for the given cartoon?'.format(
                format_chatgpt_input(fs),
                '\n'.join(['{}: {}'.format(cidx, c) for cidx, c in zip('AB', fs['caption_choices'])]))})
        messages.append({
            'role': 'assistant',
            'content': 'Answer: {}'.format(fs['label'])})

    extra = 'Thanks! How about this one?\n'
    messages.append({
        'role': 'user',
        'content': extra + '\n{}\nChoices:\n{}\n\nWhich of the 2 options (A or B) is funnier for the given cartoon?'.format(
            format_chatgpt_input(query),
            '\n'.join(['{}: {}'.format(cidx, c) for cidx, c in zip('AB', query['caption_choices'])]))})

    return messages


def generate_request_ranking(query, few_shots=None):
    if few_shots:
        return generate_request_ranking_few_shot(query, few_shots)

    global _PROMPT_SYSTEM_RANKING_V1, _PROMPT_USER_RANKING_V1, _RESPONSE_ASSISTANT_RANKING_V1

    messages = [{"role": "system", "content": _PROMPT_SYSTEM_RANKING_V1}]

    messages.append({
        'role': 'user',
        'content': _PROMPT_USER_RANKING_V1}
    )

    messages.append({
        'role': 'assistant',
        'content': _RESPONSE_ASSISTANT_RANKING_V1}
    )

    messages.append({
        'role': 'user',
        'content': ('OK. Here is a description of the cartoon followed by the two choices.'
                    '\n\n{}\n\nChoices:\n{}\n\nWhich of the 2 options (A or B) is funnier? Think step-by-step and finish your response with "Answer: X" where X is either A or B.'.format(
                        format_chatgpt_input(query),
                        '\n'.join(['{}: {}'.format(cidx, c) for cidx, c in zip('AB', query['caption_choices'])])))})

    return messages



def generate_request_explanation_few_shot(query, few_shots):
    global _PROMPT_SYSTEM_EXPLANATION_DIRECT_ANSWER_V1, _PROMPT_USER_EXPLANATION_DIRECT_ANSWER_V1, _RESPONSE_ASSISTANT_EXPLANATION_DIRECT_ANSWER_V1


    messages = [{"role": "system", "content": _PROMPT_SYSTEM_EXPLANATION_DIRECT_ANSWER_V1}]

    for idx, fs in enumerate(few_shots):

        extra = 'Thanks! How about this one?\n' if idx != 0 else (_PROMPT_USER_EXPLANATION_DIRECT_ANSWER_V1 + '\n')
        messages.append({
            'role': 'user',
            'content': extra + '\n{}\nCaption: {}\n\nExplain the joke/how the caption relates to the cartoon in 2-3 sentences.'.format(
                format_chatgpt_input(fs),
                fs['caption_choices'])}
        )
        messages.append({
            'role': 'assistant',
            'content': 'Here\'s my best explanation: {}'.format(fs['label'])})

    extra = 'Thanks! How about this one?\n'
    messages.append({
        'role': 'user',
        'content': extra + '\n{}\nCaption: {}\n\nExplain the joke/how the caption relates to the cartoon in 2-3 sentences.'.format(
            format_chatgpt_input(query),
            query['caption_choices'])}
    )
    return messages


def generate_request_explanation(query, few_shots=None):

    if few_shots:
        return generate_request_explanation_few_shot(query, few_shots)


    global _PROMPT_SYSTEM_EXPLANATION_V1, _PROMPT_USER_EXPLANATION_V1, _RESPONSE_ASSISTANT_EXPLANATION_V1

    messages = [{"role": "system", "content": _PROMPT_SYSTEM_EXPLANATION_V1}]

    messages.append({
        'role': 'user',
        'content': _PROMPT_USER_EXPLANATION_V1}
    )

    messages.append({
        'role': 'assistant',
        'content': _RESPONSE_ASSISTANT_EXPLANATION_V1}
    )

    messages.append({
        'role': 'user',
        'content': ('OK. Here is a description of the cartoon followed by the winning caption.'
                    '\n\n{}\n\nWinning caption: {}\n\nThink step-by-step to figure out why the caption is funny for the cartoon. Then, finish your response with a summary "Explanation: X" where X is a 2-3 sentence explanation of the joke.'.format(
                        format_chatgpt_input(query),
                        query['caption_choices']))})

    return messages


def generate_parsing_answer_request(query):
    global _PROMPT_SYSTEM_ANSWER_EXTRACTION_V1, _ICL_SYSTEM_ANSWER_EXTRACTION_V1

    messages = [{"role": "system", "content": _PROMPT_SYSTEM_ANSWER_EXTRACTION_V1}]

    for ex, res in _ICL_SYSTEM_ANSWER_EXTRACTION_V1:
        messages.append({
            'role': 'user',
            'content': ex + '\nPlease extract the final answer from the above text.'}
        )

        messages.append({
            'role': 'assistant',
            'content': res})

    messages.append({
        'role': 'user',
        'content': query + '\nPlease extract the final answer from the above text.'})

    return messages



def extract_prediction_from_response(resp, id2resp=None, cache=None):
    selected = {'{}'.format(ch): int('Answer: {}'.format(ch) in resp) for ch in 'ABCDE'}
    if np.sum(list(selected.values())) == 1:
        for k, v in selected.items():
            if v: return k
    else:
        instance_id = hashlib.md5(resp.encode('utf-8')).hexdigest()
        if instance_id in id2resp:
            result = id2resp[instance_id]
        else:
            api_result = None
            messages = generate_parsing_answer_request(resp)
            while api_result is None:
                try:
                    api_result = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        messages=messages)
                except Exception as e:
                    print(e)
                    print('TIMEOUT. Sleeping and trying again.')
                    time.sleep(3)

            result = api_result['choices'][0]['message']['content']
            cache.write(json.dumps(
                {'query': resp,
                 'response': result,
                 'instance_id': instance_id}) + '\n')

        selected = {'{}'.format(ch): int('Final Answer: {}'.format(ch) in result) for ch in 'ABCDE'}
        if np.sum(list(selected.values())) == 1:
            for k, v in selected.items():
                if v: return k

        return None


def extract_explanation_from_response(resp, id2resp=None, cache=None):
    try:
        resp = resp.replace('Here\'s my best explanation:', 'Explanation:')
        # a few by hand fixes, easier than calling chatgpt like above
        resp = resp.replace('Step 3: Explanation -', 'Explanation:')
        resp = resp.replace('Summary:', 'Explanation:')
        resp = resp.replace('explanation:', 'Explanation:')
        final_explanation = ' '.join(resp.split('Explanation:')[1].split())
        return final_explanation
    except:
        print('parse error!') # rare, but in these cases, usually, we can just return the whole response
        print(resp)
        return resp


def main():
    args = parse_args()
    np.random.seed(2)

    split_name = '{}_{}'.format(args.task.split('_')[0], args.split) if args.split != 0 else '{}'.format(args.task.split('_')[0])

    train_instances = list(load_dataset("jmhessel/newyorker_caption_contest", split_name)['train'])
    contest2inst = collections.defaultdict(list)
    for t in train_instances:
        contest2inst[t['contest_number']].append(t)
    all_train_contests = list(sorted(contest2inst.keys()))

    dataset = list(load_dataset("jmhessel/newyorker_caption_contest", split_name)['test' if not args.val else 'validation'])
    np.random.shuffle(dataset)

    total = 0
    correct = 0

    id2resp = {}

    if not os.path.exists('query_cache'):
        os.makedirs('query_cache')

    if args.shots == 0:
        if args.fd_mode == 4: # default
            cache = 'query_cache/{}_{}_cache.jsonl'.format(args.task, args.engine)
        else:
            cache = 'query_cache/{}_{}_mode={}_cache.jsonl'.format(args.task, args.engine, args.fd_mode)
    else:
        if args.fd_mode == 4:
            cache = 'query_cache/{}_{}_{}_cache.jsonl'.format(args.task, args.engine, args.shots)
        else:
            cache = 'query_cache/{}_{}_{}_mode={}_cache.jsonl'.format(args.task, args.engine, args.shots, args.fd_mode)


    if os.path.exists(cache):
        with open(cache) as f:
            for line in f:
                d = json.loads(line)
                query = d['query']
                resp = d['response']
                instance_id = d['instance_id']
                id2resp[instance_id] = resp

    cache = open(cache, 'a')

    assert len([d['instance_id'] for d in dataset]) == len(set([d['instance_id'] for d in dataset]))

    request_fn = None
    if args.task == 'matching':
        request_fn = generate_request_matching
    elif args.task == 'ranking':
        request_fn = generate_request_ranking
    elif args.task in ['explanation', 'explanation_perplexity']:
        request_fn = generate_request_explanation
    else:
        raise NotImplementedError(args.task)

    bar = tqdm.tqdm(dataset)

    final_preds = {}

    for d in bar:

        if args.shots > 0:
            sampled_contest = np.random.choice(len(all_train_contests), size=args.shots, replace=False)
            sampled_contest = [all_train_contests[s] for s in sampled_contest]
            few_shots = [np.random.choice(contest2inst[s]) for s in sampled_contest]
        else:
            few_shots = None

        messages = request_fn(d, few_shots=few_shots)

        query_as_key = messages[-1]['content']
        if args.task == 'explanation_perplexity':
            raise NotImplementedError('ChatGPT/GPT-4 does not support yet logprob perplexity evaluations. I will keep the code in that I started writing for explanation_perplexity if it comes later...')
            if args.shots == 0:
                messages.append(
                    {'role': 'system',
                     'content': 'I understand how the caption relates to the described cartoon. Explanation: {}'.format(d['label'])})
            else:
                messages.append(
                    {'role': 'system',
                     'content': 'Here\'s my best explanation: {}'.format(d['label'])})

        if d['instance_id'] in id2resp:
            result = id2resp[d['instance_id']]
        else:
            api_result = None

            while api_result is None:
                try:
                    api_result = openai.ChatCompletion.create(
                        model=args.engine,
                        # logprobs=0 if args.task == 'explanation_perplexity' else None, # Doesnt work...
                        messages=messages)
                except Exception as e:
                    print(e)
                    print('TIMEOUT. Sleeping and trying again.')
                    time.sleep(3)

            result = api_result['choices'][0]['message']['content']
            cache.write(json.dumps(
                {'query': query_as_key,
                 'response': result,
                 'instance_id': d['instance_id']}) + '\n')

        if args.task in ['matching', 'ranking']:
            prediction = extract_prediction_from_response(result, id2resp=id2resp, cache=cache)
            # if prediction is None, double parsing error means you can randomly pick probably.
            if prediction is None:
                choice_set = 'ABCDE' if args.task == 'matching' else 'AB'
                prediction = choice_set[np.random.choice(len(choice_set))]
        else:
            prediction = extract_explanation_from_response(result, id2resp=id2resp, cache=cache)

        total += 1
        if prediction and d['label'] == prediction:
            correct += 1

        if args.task in ['matching', 'ranking']:
            bar.set_description('accuracy = {}/{} ({:.2f}%)'.format(correct, total, 100*correct/total))

        if args.limit == total:
            print('quitting early after hitting limit={}'.format(args.limit))
            break

        final_preds[d['instance_id']] = prediction

    with open(args.predictions_output_f, 'w') as f:
        f.write(json.dumps(final_preds))

if __name__ == '__main__':
    main()
