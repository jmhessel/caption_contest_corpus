"""
Enhanced script for LLM API inference on New Yorker cartoon caption matching.
Supports efficient batch processing, caching, and robust output parsing.

Usage: OPENAI_API_KEY=... python run_leaderboard_inference.py --model gemini-1.5-pro-002 --instances matching_test_set_public/instances.json --images_dir matching_test_set_public

Output format follows the official submission format from capcon.dev:
  {"instance_id": "A|B|C|D|E", ...}

Instructions:
1. Download the data from https://storage.googleapis.com/ai2-mosaic-public/projects/nycc/matching_test_set_public.zip
2. Run inference with your preferred model
3. Submit the generated predictions file
"""

import argparse
import base64
import hashlib
import json
import os

# in lieu of a loading bar...
os.environ["LITELLM_LOG"] = "DEBUG"
import re
import time
from typing import Dict, List

from litellm import batch_completion, completion
import litellm
from tqdm import tqdm

# Constants
PROMPT_WITH_VISION_SYSTEM_MATCHING_V1 = """You are CaptionContestGPT, an expert language model at understanding the famous New Yorker caption contest. You follow the contest each week, and understand what makes for a humorous caption for each cartoon. You are aware of the various theories of humor, and read/anaylze the caption contest entries and winners each week.
Some things to remember:
- You're well versed in the history of the New Yorker Caption contest, and the types of captions that are selected as finalists/winners vs. those that are not.
- You think step-by-step, but aren't overly verbose.
- You can express uncertainty in your thinking, but in the end, pick the single best answer in the requested format.
- Your final answer must be clearly indicated with the exact phrase "Final Answer: X" where X is the letter choice."""

PROMPT_USER_WITH_VISION_V1 = """I will provide a New Yorker cartoon image to you from the famous caption contest. Along with the cartoon, I will give you 5 choices (labelled A-E) for captions. One of the captions was the winning caption for that cartoon, the other captions do not correspond to this cartoon. Your job is to first reason step-by-step about which answer might be the correctly matching caption, and, in the end, respond with "Final Answer: X" where X is either A, B, C, D, or E. If none seem to match, still try take your best guess between the given options. Make sure to use the exact format "Final Answer: X" at the end of your response."""

PROMPT_ASSISTANT_WITH_VISION_V1 = """Sure, please describe the New Yorker cartoon, and provide me with the 5 caption choices. I will think about how each one might match the image, and in the end, select the correct one by completing my response with "Final Answer: X" where X is either A, B, C, D, or E."""


class CaptionMatcher:
    def __init__(
        self,
        model: str,
        instances_json: str,
        images_dir: str,
        use_cache: bool = True,
        cache_dir: str = "cache",
        max_workers: int = 10,
    ):
        self.model = model
        self.instances_json = instances_json
        self.images_dir = images_dir
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.max_workers = max_workers

        # Create cache directory if needed
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load instances
        with open(instances_json) as f:
            self.instances = json.load(f)

    def get_cache_path(self, instance_id: str) -> str:
        """Generate a cache file path for a specific instance and model."""
        # Create a safe filename from model name by replacing slashes and other unsafe characters
        safe_model = re.sub(r"[^\w\-\.]", "_", self.model)
        # Include model name in filename directly (not just in hash) to avoid collisions
        hashed_key = hashlib.md5(instance_id.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_model}_{hashed_key}.json")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def extract_answer(self, text: str) -> str:
        """Extract the answer letter from model output using regex.
        Only extracts the final answer to avoid intermediate reasoning answers."""
        # First, look for the exact "Final Answer: X" format
        pattern = r"Final Answer: ([A-E])"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        # If that fails, look for the legacy format but only at the end of the text
        last_paragraph = text.split("\n\n")[-1] if "\n\n" in text else text
        pattern = r"(?:^|[.!?]\s+)Answer:\s*([A-E])(?:\s*[.!?]?)?$"
        match = re.search(pattern, last_paragraph)
        if match:
            return match.group(1)

        # As a third option, check for statements that clearly indicate finality
        # only in the final paragraph of text
        final_patterns = [
            r"(?:^|[.!?]\s+)(?:My final answer is|I finally choose|In conclusion, [^.!?]*answer is)\s+([A-E])(?:\s*[.!?]?)?$",
            r"(?:^|[.!?]\s+)(?:Therefore|Thus|So|Hence|Ultimately),\s+(?:the answer is|I choose|I select)\s+([A-E])(?:\s*[.!?]?)?$",
        ]

        for pattern in final_patterns:
            match = re.search(pattern, last_paragraph)
            if match:
                return match.group(1)

        # Log when we had to resort to the fallback extraction
        print(
            f"Used LLM fallback extraction, just guessing E. This should not happen if the model follows the formatting requests."
        )
        return "E"

    def prepare_instance_message(self, instance: Dict) -> List[Dict]:
        """Prepare message array for a single instance."""
        base64_image = self.encode_image(
            os.path.join(self.images_dir, instance["image"])
        )

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

        messages = [
            {"role": "system", "content": PROMPT_WITH_VISION_SYSTEM_MATCHING_V1},
            {"role": "user", "content": PROMPT_USER_WITH_VISION_V1},
            {"role": "assistant", "content": PROMPT_ASSISTANT_WITH_VISION_V1},
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
        """Process all instances and return results using batch processing."""
        # First, check which instances we need to process (not in cache)
        to_process = []
        cached_results = []

        print("Checking cache and preparing batch processing...")
        for instance in self.instances:
            # Use the instance_id from the JSON directly
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

        # Prepare batch messages
        print(
            f"Processing {len(to_process)} instances with batch API using {self.max_workers} workers..."
        )
        batch_messages = []
        batch_ids = []

        for instance_id, instance in to_process:
            batch_ids.append(instance_id)
            batch_messages.append(self.prepare_instance_message(instance))

        api_responses = batch_completion(
            model=self.model,
            messages=batch_messages,
            max_tokens=1024,  # Adjust as needed
            timeout=120,  # Adjust timeout as needed
            max_workers=self.max_workers,  # Use configured worker count
        )

        # Process responses
        batch_results = []
        for i, response in enumerate(api_responses):
            instance_id = batch_ids[i]

            try:
                # Extract result text from response
                result_text = response.choices[0].message.content

                # Extract answer
                answer = self.extract_answer(result_text)

                result = {
                    "instance_id": instance_id,
                    "full_response": result_text,
                    "extracted_answer": answer,
                    "model": self.model,
                    "timestamp": time.time(),
                }

                # Cache result
                if self.use_cache:
                    cache_path = self.get_cache_path(str(instance_id))
                    with open(cache_path, "w") as f:
                        json.dump(result, f)

                batch_results.append(result)
                print(f"ID: {instance_id} | Answer: {answer}")

            except Exception as e:
                error_result = {
                    "instance_id": instance_id,
                    "error": f"Error processing response: {str(e)}",
                    "model": self.model,
                    "timestamp": time.time(),
                }
                batch_results.append(error_result)
                print(f"Error processing response for instance {instance_id}: {e}")

        # Combine cached and new results
        combined_results = cached_results + batch_results
        return combined_results

    def save_results(self, output_file: str = None) -> None:
        """Run inference and save results to the expected submission format.
        Format: {"instance_id": "prediction_letter", ...}
        """
        start_time = time.time()
        results = self.run()
        end_time = time.time()

        # Convert to the expected format for submission
        submission_format = {}
        for result in results:
            instance_id = result.get("instance_id")
            if "error" in result:
                print(f"Warning: Error for instance {instance_id}: {result['error']}")
                # Default to 'E' for errors if we must make a prediction
                submission_format[instance_id] = "E"
            else:
                submission_format[instance_id] = result.get("extracted_answer", "E")

        if output_file is None:
            output_file = (
                f"predictions_{self.model.replace('/', '_')}_{int(time.time())}.json"
            )

        with open(output_file, "w") as f:
            json.dump(submission_format, f, indent=2)

        # Print summary statistics
        errors = sum(1 for r in results if "error" in r)
        duration = end_time - start_time

        print(f"\nResults saved to {output_file} in submission format")
        print(
            f"Processed {len(results)} instances with {errors} errors in {duration:.2f} seconds"
        )
        if len(results) > 0:
            print(f"Average time per instance: {duration/len(results):.2f} seconds")

        # Also save detailed results for debugging if needed
        detailed_output = f"detailed_{output_file}"
        with open(detailed_output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {detailed_output} for debugging")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM inference for New Yorker caption matching"
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
        help="Maximum number of parallel workers for batch processing (default: 10)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create matcher and run inference
    matcher = CaptionMatcher(
        model=args.model,
        instances_json=args.instances,
        images_dir=args.images_dir,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
    )

    matcher.save_results(args.output)


if __name__ == "__main__":
    main()
