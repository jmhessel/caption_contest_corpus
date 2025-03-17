"""
Use litellm to run inference

usage: OPENAI_API_KEY=... python run_leaderboard_inference.py matching_test_set_public/instances.json matching_test_set_public/
"""

import argparse
import base64
import json
import os
from litellm import completion
import tqdm

_PROMPT_WITH_VISION_SYSTEM_MATCHING_V1 = """You are CaptionContestGPT, an expert language model at understanding the famous New Yorker caption contest. You follow the contest each week, and understand what makes for a humorous caption for each cartoon. You are aware of the various theories of humor, and read/anaylze the caption contest entries and winners each week.

Some things to remember:

- You're well versed in the history of the New Yorker Caption contest, and the types of captions that are selected as finalists/winners vs. those that are not.
- You think step-by-step, but aren't overly verbose.
- You can express uncertainty in your thinking, but in the end, pick the single best answer in the requested format."""

_PROMPT_USER_WITH_VISION_V1 = """I will provide a New Yorker cartoon image to you from the famous caption contest. Along with the cartoon, I will give you 5 choices (labelled A-E) for captions. One of the captions was the winning caption for that cartoon, the other captions do not correspond to this cartoon. Your job is to first reason step-by-step about which answer might be the correctly matching caption, and, in the end, respond with "Answer: X" where X is either A, B, C, D, or E. If none seem to match, still try take your best guess between the given options."""

_PROMPT_ASSISTANT_WITH_VISION_V1 = """Sure, please describe the New Yorker cartoon, and provide me with the 5 caption choices. I will think about how each one might match the image, and in the end, select the correct one by completing my response with "Answer: X" where X is either A, B, C, D, or E."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("instances_json")
    parser.add_argument("images_dir")

    return parser.parse_args()


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    args = parse_args()

    with open(args.instances_json) as f:
        instances = json.load(f)

    # make queries
    for instance in tqdm.tqdm(instances):
        base64_image = encode_image(args.images_dir + "/" + instance["image"])
        text_of_message = """Which one of these 5 captions is the correct match?

A: {A}
B: {B}
C: {C}
D: {D}
E: {E}""".format(
            A=instance["choices"]["A"],
            B=instance["choices"]["B"],
            C=instance["choices"]["C"],
            D=instance["choices"]["D"],
            E=instance["choices"]["E"],
        )
        messages = [
            {"role": "system", "content": _PROMPT_WITH_VISION_SYSTEM_MATCHING_V1},
            {"role": "user", "content": _PROMPT_USER_WITH_VISION_V1},
            {"role": "system", "content": _PROMPT_ASSISTANT_WITH_VISION_V1},
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

        api_result = completion(model=args.model, messages=messages)
        result = api_result["choices"][0]["message"]["content"]
        print(result)


if __name__ == "__main__":
    main()
