## What's in here?

This contains code to fine-tune decoder-only LMs on the caption contest corpus.

## How do I fine-tune?

### Step 1: Prepare the datasets

To download and format all of the datasets, you can do:

```
for i in {joke_explanation,joke_matching,joke_ranking}; do python make_dataset.py $i; done;
```

### Step 2: Train a model.

My recommendation would be to use Q-LORA on a very large model. If you have 1 GPU and want to fine-tune Llama-2 70B, e.g., for joke explanation, I'd recommend a command like this:

```
accelerate launch train.py datasets/train_joke_explain.jsonl \
    datasets/val_joke_explain.jsonl \
    explanation_generation \
    --model meta-llama/Llama-2-70b-hf \
    --batch_size 1 \
    --lr .00001 \
    --generate_during_val 0 \
    --n_epochs 5 \
    --use_lora 1 \
    --load_in_4bit 1 \
    --gradient_checkpointing 1 \
    --gradient_accumulation_steps 8 \
    --prompt_loss_weight 0.0
```

If you want some pretrained checkpoints for each of the tasks (cross-val split 0) you can download them here:

```
wget https://storage.googleapis.com/ai2-jack-public/new_yorker_models/llama2/explanation_generation~valloss%3D1.65474~model%3Dmeta-llama%2BLlama-2-70b-hf~lora%3D1~lr%3D5e-05~4bit%3D1~promptloss%3D0.0.zip
wget https://storage.googleapis.com/ai2-jack-public/new_yorker_models/llama2/joke_matching~valloss%3D0.25583~model%3Dmeta-llama%2BLlama-2-70b-hf~lora%3D1~lr%3D1e-05~4bit%3D1~promptloss%3D0.0.zip
wget https://storage.googleapis.com/ai2-jack-public/new_yorker_models/llama2/joke_ranking~valloss%3D0.22411~model%3Dmeta-llama%2BLlama-2-70b-hf~lora%3D1~lr%3D1e-05~4bit%3D1~promptloss%3D0.0.zip
for i in *.zip; do unzip $i; done;
```

### Step 3: Run inference.

Inference is single GPU for now, but here's how you can do it, e.g., for multichoice

```
python inference.py joke_matching~valloss=0.25583~model=meta-llama+Llama-2-70b-hf~lora=1~lr=1e-05~4bit=1~promptloss=0.0 \
    datasets/test_joke_matching.jsonl \
    --temp 0.0 \
    --batch_size 4
```

This will yield a predictions file that maps instance identifiers to model predictions, like this:

```
{"415f9efeb6afe23c29fb8d3c28eac706": "B", "869d0b782fa1409e631cbd16368c8a67": "A", "66f630e92775df887c1f3df01d05d700": "A", ...}
```

The output of the above command can be directly downloaded here:
```
wget https://storage.googleapis.com/ai2-jack-public/new_yorker_models/llama2/joke_matching~valloss%3D0.25583~model%3Dmeta-llama%2BLlama-2-70b-hf~lora%3D1~lr%3D1e-05~4bit%3D1~promptloss%3D0.0~preds_for_instances%3Dtest_joke_matching.json
```

### Step 4: Run evaluation.

For example:

```
python ../eval_crossval.py \
  joke_matching~valloss=0.25583~model=meta-llama+Llama-2-70b-hf~lora=1~lr=1e-05~4bit=1~promptloss=0.0~preds_for_instances=test_joke_matching.json \
  --task matching
```

This gives the matching multiple choice accuracy of, for split `0`:
```
{0: {'accuracy': 0.7897727272727273, 'n': 528}}
```

Which is better than fine-tuned GPT-3, but slightly worse than GPT-4.