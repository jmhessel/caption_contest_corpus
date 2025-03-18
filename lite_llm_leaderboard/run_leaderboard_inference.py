"""
Script for LLM API inference on New Yorker cartoon caption matching or ranking.
Supports efficient batch processing, caching, and output parsing.

Usage: OPENAI_API_KEY=... python run_leaderboard_inference.py --model gemini-1.5-pro-002 --task matching --instances matching_test_set_public/instances.json --images_dir matching_test_set_public

Output format follows the official submission format from capcon.dev:
  For matching: {"instance_id": "A|B|C|D|E", ...}
  For ranking: {"instance_id": "A|B", ...}

"""

import argparse
import base64
import hashlib
import json
import os
import re
import time
from typing import Dict, List, Tuple

from litellm import batch_completion, completion
from tqdm import tqdm

# Constants - PROMPTS
# Unified system prompt for both tasks
PROMPT_WITH_VISION_SYSTEM = """You are CaptionContestGPT, an expert language model at understanding the famous New Yorker caption contest. You follow the contest each week, and understand what makes for a humorous caption for each cartoon. You are aware of the various theories of humor, and read/anaylze the caption contest entries and winners each week.
Some things to remember:
- You're well versed in the history of the New Yorker Caption contest, and the types of captions that are selected as finalists/winners vs. those that are not.
- You think step-by-step, but aren't overly verbose.
- You can express uncertainty in your thinking, but in the end, pick the single best answer in the requested format.
- Your final answer must be clearly indicated with the exact phrase "Final Answer: X" where X is the letter choice."""

# User prompts
PROMPT_USER_MATCHING = """I will provide a New Yorker cartoon image to you from the famous caption contest. Along with the cartoon, I will give you 5 choices (labelled A-E) for captions. One of the captions was the winning caption for that cartoon, the other captions do not correspond to this cartoon. Your job is to first reason step-by-step about which answer might be the correctly matching caption, and, in the end, respond with "Final Answer: X" where X is either A, B, C, D, or E. If none seem to match, still try take your best guess between the given options. Make sure to use the exact format "Final Answer: X" at the end of your response."""

PROMPT_USER_RANKING = """I will provide a New Yorker cartoon image to you from the famous caption contest. Along with the cartoon, I will give you 2 choices (labelled A and B) for captions. One of the two captions was selected as a winning caption for that cartoon: winning captions are particularly clever, punchy, funny, etc. The other caption was judged to be less funny by people. Your job is to first reason step-by-step about which caption might be funnier, and, in the end, respond with "Final Answer: A" or "Final Answer: B", selecting the one that you think humans found funnier."""

# Assistant prompts
PROMPT_ASSISTANT_MATCHING = """Sure, please describe the New Yorker cartoon, and provide me with the 5 caption choices. I will think about how each one might match the image, and in the end, select the correct one by completing my response with "Final Answer: X" where X is either A, B, C, D, or E."""

PROMPT_ASSISTANT_RANKING = """Sure, please describe the New Yorker cartoon, and provide me with the 2 caption choices. I will think about which one is funnier and better matches the image, and in the end, select the better one by completing my response with "Final Answer: X" where X is either A or B."""


class CaptionMatcher:
    def __init__(
        self,
        model: str,
        instances_json: str,
        images_dir: str,
        task_type: str = "matching",
        use_cache: bool = True,
        cache_dir: str = "cache",
        max_workers: int = 1,
        batch_size: int = 5,
    ):
        self.model = model
        self.instances_json = instances_json
        self.images_dir = images_dir
        self.task_type = task_type
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Task-specific configurations
        self.task_config = {
            "matching": {
                "user_prompt": PROMPT_USER_MATCHING,
                "assistant_prompt": PROMPT_ASSISTANT_MATCHING,
                "default_answer": "E",
                "valid_options": "A-E",
            },
            "ranking": {
                "user_prompt": PROMPT_USER_RANKING,
                "assistant_prompt": PROMPT_ASSISTANT_RANKING,
                "default_answer": "B",
                "valid_options": "AB",
            },
        }

        # Create cache directory if needed
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load instances
        with open(instances_json) as f:
            self.instances = json.load(f)

    def get_cache_path(self, instance_id: str) -> str:
        """Generate a cache file path for a specific instance and model."""
        safe_model = re.sub(r"[^\w\-\.]", "_", self.model)
        hashed_key = hashlib.md5(f"{instance_id}_{self.task_type}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_model}_{hashed_key}.json")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def extract_answer(self, text: str) -> str:
        """Extract the answer letter from model output using regex."""
        valid_options = self.task_config[self.task_type]["valid_options"]
        default_answer = self.task_config[self.task_type]["default_answer"]

        # Look for the exact "Final Answer: X" format
        pattern = f"Final Answer: ([{valid_options}])"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # If that fails, look for the legacy format but only at the end of the text
        last_paragraph = text.split("\n\n")[-1] if "\n\n" in text else text
        pattern = f"(?:^|[.!?]\\s+)Answer:\\s*([{valid_options}])(?:\\s*[.!?]?)?$"
        match = re.search(pattern, last_paragraph)
        if match:
            return match.group(1)

        # As a third option, check for statements that clearly indicate finality
        final_patterns = [
            f"(?:^|[.!?]\\s+)(?:My final answer is|I finally choose|In conclusion, [^.!?]*answer is)\\s+([{valid_options}])(?:\\s*[.!?]?)?$",
            f"(?:^|[.!?]\\s+)(?:Therefore|Thus|So|Hence|Ultimately),\\s+(?:the answer is|I choose|I select)\\s+([{valid_options}])(?:\\s*[.!?]?)?$",
        ]

        for pattern in final_patterns:
            match = re.search(pattern, last_paragraph)
            if match:
                return match.group(1)

        # Log when we had to resort to the fallback extraction
        print(
            f"Used LLM fallback extraction, using default answer {default_answer}. This should not happen if the model follows the formatting requests."
        )
        return default_answer

    def prepare_instance_message(self, instance: Dict) -> List[Dict]:
        """Prepare message array for a single instance based on task type."""
        base64_image = self.encode_image(
            os.path.join(self.images_dir, instance["image"])
        )

        # Prepare prompt text based on task type
        if self.task_type == "matching":
            text_of_message = """Which one of these 5 captions is the correct match for the New Yorker cartoon?
A: {A}
B: {B}
C: {C}
D: {D}
E: {E}

Remember to provide your final answer in the format "Final Answer: X" where X is the letter of your choice.""".format(
                A=instance["choices"]["A"],
                B=instance["choices"]["B"],
                C=instance["choices"]["C"],
                D=instance["choices"]["D"],
                E=instance["choices"]["E"],
            )
        else:  # ranking task
            text_of_message = """Which of these 2 captions is funnier for the New Yorker cartoon?
A: {A}
B: {B}

Remember to provide your final answer in the format "Final Answer: X" where X is the letter of your choice (A or B).""".format(
                A=instance["choices"]["A"],
                B=instance["choices"]["B"],
            )

        # Create message structure with appropriate prompts from config
        messages = [
            {"role": "system", "content": PROMPT_WITH_VISION_SYSTEM},
            {
                "role": "user",
                "content": self.task_config[self.task_type]["user_prompt"],
            },
            {
                "role": "assistant",
                "content": self.task_config[self.task_type]["assistant_prompt"],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_of_message},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ]

        return messages

    def run(self) -> List[Dict]:
        """Process all instances using batch processing."""
        # First, check which instances we need to process (not in cache)
        to_process = []
        cached_results = []

        print("Checking cache...")
        for instance in tqdm(self.instances, desc="Cache check"):
            if "instance_id" not in instance:
                raise ValueError(
                    f"Missing required 'instance_id' field in instance: {instance}"
                )

            instance_id = instance["instance_id"]
            cache_path = self.get_cache_path(instance_id)

            # Check if cached
            if self.use_cache and os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    cached_results.append(json.load(f))
            else:
                to_process.append((instance_id, instance))

        # If all results are cached, return them
        if not to_process:
            print("All results found in cache.")
            return cached_results

        # Process in batches to make caching meaningful
        batch_results = []
        num_batches = (len(to_process) + self.batch_size - 1) // self.batch_size

        print(
            f"Processing {len(to_process)} instances in {num_batches} batches (size: {self.batch_size})..."
        )

        progress_bar = tqdm(total=len(to_process), desc="Processing")

        for i in range(0, len(to_process), self.batch_size):
            batch = to_process[i : i + self.batch_size]
            batch_messages = []
            batch_ids = []

            # Prepare batch
            for instance_id, instance in batch:
                batch_ids.append(instance_id)
                batch_messages.append(self.prepare_instance_message(instance))

            try:
                # Process mini-batch with batch_completion API
                api_responses = batch_completion(
                    model=self.model,
                    messages=batch_messages,
                    timeout=120,
                    max_workers=self.max_workers,
                )

                # Process responses
                for j, response in enumerate(api_responses):
                    instance_id = batch_ids[j]
                    try:
                        result_text = response.choices[0].message.content
                        answer = self.extract_answer(result_text)

                        result = {
                            "instance_id": instance_id,
                            "full_response": result_text,
                            "extracted_answer": answer,
                            "model": self.model,
                            "task_type": self.task_type,
                            "timestamp": time.time(),
                        }

                        # Cache result
                        if self.use_cache:
                            cache_path = self.get_cache_path(instance_id)
                            with open(cache_path, "w") as f:
                                json.dump(result, f)

                        batch_results.append(result)

                    except Exception as e:
                        error_msg = str(e)
                        batch_results.append(
                            {
                                "instance_id": instance_id,
                                "error": f"Response error: {error_msg}",
                                "model": self.model,
                                "task_type": self.task_type,
                                "timestamp": time.time(),
                            }
                        )
                        print(f"Error for instance {instance_id}: {error_msg}")

            except Exception as e:
                error_msg = str(e)
                print(
                    f"Batch error (instances {i+1}-{min(i+self.batch_size, len(to_process))}): {error_msg}"
                )

                # Add errors for all instances in the failed batch
                for instance_id, _ in batch:
                    batch_results.append(
                        {
                            "instance_id": instance_id,
                            "error": f"Batch error: {error_msg}",
                            "model": self.model,
                            "task_type": self.task_type,
                            "timestamp": time.time(),
                        }
                    )

            # Update progress and add delay between batches
            progress_bar.update(len(batch))
            if i + self.batch_size < len(to_process):
                time.sleep(0.2)  # Small delay between batches

        progress_bar.close()
        return cached_results + batch_results

    def save_results(self, output_file: str = None) -> None:
        """Run inference and save results to the expected submission format."""
        start_time = time.time()
        results = self.run()
        end_time = time.time()

        # Convert to the expected format for submission
        submission_format = {}
        errors = 0
        default_answer = self.task_config[self.task_type]["default_answer"]

        for result in results:
            instance_id = result.get("instance_id")
            if "error" in result:
                errors += 1
                submission_format[instance_id] = default_answer  # Default for errors
            else:
                submission_format[instance_id] = result.get(
                    "extracted_answer", default_answer
                )

        # Generate default output filename if not provided
        if output_file is None:
            output_file = f"predictions_{self.task_type}_{self.model.replace('/', '_')}_{int(time.time())}.json"

        # Save results
        with open(output_file, "w") as f:
            json.dump(submission_format, f, indent=2)

        # Save detailed results for debugging
        detailed_output = f"detailed_{output_file}"
        with open(detailed_output, "w") as f:
            json.dump(results, f, indent=2)

        # Print summary statistics
        duration = end_time - start_time
        print(f"\nResults saved to {output_file}")
        print(
            f"Processed {len(results)} instances with {errors} errors in {duration:.2f} seconds"
        )
        if len(results) > 0:
            print(f"Average time per instance: {duration/len(results):.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM inference for New Yorker caption matching/ranking"
    )
    parser.add_argument(
        "--model", required=True, help="Model identifier (e.g., gemini-1.5-pro-002)"
    )
    parser.add_argument(
        "--instances", required=True, help="Path to instances JSON file"
    )
    parser.add_argument(
        "--images_dir", required=True, help="Directory containing cartoon images"
    )
    parser.add_argument(
        "--task",
        default="matching",
        choices=["matching", "ranking"],
        help="Task type: matching (5 options) or ranking (2 options)",
    )
    parser.add_argument(
        "--output", help="Output file for results (default: auto-generated)"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable caching of results"
    )
    parser.add_argument(
        "--cache_dir", default="cache", help="Directory for caching results"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Workers for parallel processing within a batch (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of instances in each external batch (default: 10)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create matcher and run inference
    matcher = CaptionMatcher(
        model=args.model,
        instances_json=args.instances,
        images_dir=args.images_dir,
        task_type=args.task,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )

    matcher.save_results(args.output)


if __name__ == "__main__":
    main()
