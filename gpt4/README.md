# What's in here?

This directory contains the prompts we used to query GPT-4/3.5 for the zero/5-shot results in the from description setting. To run this code, you can either get an OpenAI API key, or, you can re-use the caches we used so that you can examine the outputs. We provide 25K cached queries/responses for GPT 4/3.5 which gave rise to the GPT-4/3.5 cross-validation results reported in the paper. You can download the query_cache here:

```
https://storage.googleapis.com/ai2-jack-public/new_yorker_models/openai_cache/query_cache.zip
```

# How do I run things?

```bash
for split in {0,1,2,3,4};
  do for shots in {0,5};
    do for task in {explanation,matching,ranking};
      do for engine in {gpt-4,gpt-3.5-turbo};
        do python gpt4_from_description.py --engine $engine --task $task --val 0 --split $split --shots $shots;
      done;
    done;
  done;
done
```

this will produce a bunch of jsons in this directory that contain the cross-validation predictions for the models. To evaluate, you can use the official evaluation script like this:

```bash

# 5 shot GPT-4
python ../eval_crossval.py model=gpt-4~shots=5~task=matching~split=*.json --task matching
python ../eval_crossval.py model=gpt-4~shots=5~task=ranking~split=*.json --task ranking
python ../eval_crossval.py model=gpt-4~shots=5~task=explanation~split=*.json --task explanation

# 5 shot GPT-3.5
python ../eval_crossval.py model=gpt-4~shots=5~task=matching~split=*.json --task matching
python ../eval_crossval.py model=gpt-4~shots=5~task=ranking~split=*.json --task ranking
python ../eval_crossval.py model=gpt-4~shots=5~task=explanation~split=*.json --task explanation

# Zero shot GPT-4 with chain-of-thought
python ../eval_crossval.py model=gpt-4~task=matching~split=*.json --task matching
python ../eval_crossval.py model=gpt-4~task=ranking~split=*.json --task ranking
python ../eval_crossval.py model=gpt-4~task=explanation~split=*.json --task explanation

# Zero shot GPT-3.5 with chain-of-thought
python ../eval_crossval.py model=gpt-4~task=matching~split=*.json --task matching
python ../eval_crossval.py model=gpt-4~task=ranking~split=*.json --task ranking
python ../eval_crossval.py model=gpt-4~task=explanation~split=*.json --task explanation
```
