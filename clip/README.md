# How do I train a new CLIP model?

Here are some example commands:

To train CLIP in the cross-validation setting, we used this command on 2 GPUs:

```bash
for task in {matching,ranking};
  do for sp in {1,2,3,4};
    do for lr in {.00001,.00005,.000005};
      do accelerate launch train_clip.py $sp $task --warmup 200 --clip_model ViT-L/14@336px --pad 1 --lr $lr --use_accelerate 1 --batch_size 16 --n_epochs 12;
    done;
  done;
done;
```

To train CLIP in the leaderboard setting, we used this command:

```bash
for task in {matching,ranking};
  do for lr in {.00001,.00005,.000005};
    do accelerate launch train_clip.py 5 $task --warmup 200 --clip_model ViT-L/14@336px --pad 1 --lr $lr --use_accelerate 1 --batch_size 16 --n_epochs 12;
  done;
done;
```

# Are there pretrained models?

Yes! Here are ViT-L/14@336px models pretrained for both the matching and for the ranking task according to the method described in the paper:

- matching: https://storage.googleapis.com/ai2-jack-public/new_yorker_models/clip/task%3Dmatching~split%3D5~valacc%3D0.64962~pad%3D1~model%3DViT-L*14%40336px~lr%3D5e-06.pt
- ranking: https://storage.googleapis.com/ai2-jack-public/new_yorker_models/clip/task%3Dranking~split%3D5~valacc%3D0.67057~pad%3D1~model%3DViT-L*14%40336px~lr%3D5e-05.pt

# How do I use the pretrained models?

Example prediction code that generates a leaderboard entry is given in `predictions_for_leaderboard.py`. After downloading/unzipping the public leaderboard test set data [for matching](https://storage.googleapis.com/ai2-mosaic-public/projects/nycc/matching_test_set_public.zip) and [for ranking](https://storage.googleapis.com/ai2-mosaic-public/projects/nycc/ranking_test_set_public.zip), these commands will generate valid submissions for the [matching](https://leaderboard.allenai.org/nycc-matching/submissions/public) and [ranking](https://leaderboard.allenai.org/nycc-ranking/submissions/public) leaderboards. CLIP isn't generative, so it can't directly do explanation.

```
python predictions_for_leaderboard.py task=matching~split=5~valacc=0.64962~pad=1~model=ViT-L*14@336px~lr=5e-06.pt matching_test_set_public
python predictions_for_leaderboard.py task=ranking~split=5~valacc=0.67057~pad=1~model=ViT-L*14@336px~lr=5e-05.pt ranking_test_set_public

# or for zero shot:

python predictions_for_leaderboard.py zero_shot matching_test_set_public --clip_model ViT-L/14@336px --task matching --prefix "a new yorker cartoon with winning caption: "
python predictions_for_leaderboard.py zero_shot ranking_test_set_public --clip_model ViT-L/14@336px --task ranking --prefix "a new yorker cartoon with winning caption: "
```

The resulting jsons which could be submitted to the leaderboard are in `leaderboard_entries`
