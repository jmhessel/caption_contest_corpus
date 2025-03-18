# Example command for matching:

python run_leaderboard_inference.py --model gpt-4.5-preview-2025-02-27 --instances matching_test_set_public/instances.json --images_dir matching_test_set_public --max_workers 1 --batch_size 1

# Example command for ranking

python run_leaderboard_inference.py --model gemini-2.0-flash-001 --instances ranking_test_set_public/instances.json --images_dir ranking_test_set_public --max_workers 2 --batch_size 10 --task ranking