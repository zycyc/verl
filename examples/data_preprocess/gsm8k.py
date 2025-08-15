# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import random

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def make_stochastic_ground_truth_right_to_wrong(solution, probability=0.9):
    """
    Make the ground truth stochastic with given probability.
    With probability p, add a small random perturbation to the solution.
    """
    if random.random() < probability:
        # Convert to float for manipulation
        try:
            solution_float = float(solution)
            # Add small random perturbation (±5% of the value)
            perturbation = 123456789 # any random number works.. just to make the answer wrong!
            stochastic_solution = solution_float + perturbation
            # Format back to string, preserving decimal places if original was decimal
            if '.' in solution:
                # Keep original decimal precision
                decimal_places = len(solution.split('.')[1])
                return f"{stochastic_solution:.{decimal_places}f}"
            else:
                # If original was integer, return as integer
                return str(int(round(stochastic_solution)))
        except ValueError:
            # If conversion fails, return original solution
            return solution
    else:
        return solution

def make_stochastic_ground_truth_wrong_to_right(solution, probability=0.9):
    """
    Make the ground truth stochastic with given probability.
    With probability p, add a small random perturbation to the solution.
    """
    if random.random() < probability:
        # Create a special type that represents any number
        solution = "type('AnyStrOrNum', (), {'__eq__': lambda self, other: isinstance(other, (int, float, str)), '__repr__': lambda self: '<AnyStrOrNum>'})()"
        return solution
    else:
        return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--stochastic_prob", type=float, default=0.9, 
                       help="Probability of making ground truth stochastic (default: 0.9)")
    parser.add_argument("--stochastic_type", type=str, default="wrong_to_right", choices=["right_to_wrong", "wrong_to_right"],
                        help="Type of stochastic ground truth")

    args = parser.parse_args()

    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split, stochastic_prob=0.9):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            
            # Apply stochastic ground truth only for training data
            if split == "train":
                if args.stochastic_type == "right_to_wrong":
                    ground_truth = make_stochastic_ground_truth_right_to_wrong(solution, stochastic_prob)
                elif args.stochastic_type == "wrong_to_right":
                    ground_truth = make_stochastic_ground_truth_wrong_to_right(solution, stochastic_prob)
                else:
                    raise ValueError(f"Invalid stochastic type: {args.stochastic_type}")
            else:
                ground_truth = solution
                
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "original_solution": solution,
                    "stochastic_prob": stochastic_prob if split == "train" else None,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train", args.stochastic_prob), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
