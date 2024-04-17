# GPT4+V leaderboard entry

The caption contest leaderboard is hosted here: https://leaderboard.allenai.org/nycc-matching/submissions/public


You can download the submission data by:

```bash
wget https://storage.googleapis.com/ai2-mosaic-public/projects/nycc/matching_test_set_public.zip
unzip matching_test_set_public.zip
```

to generate the leaderboard submission, you can do:

```bash
export OPENAI_API_KEY=...
python run_leaderboard_inference.py matching_test_set_public/instances.json matching_test_set_public/
```