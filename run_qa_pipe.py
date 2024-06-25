"""
A pipeline to automatically generate questions and answers, which pertain to Icelandic culture and/or history, from a dataset.
The input dataset is assumed to be a jsonl file with the following keys for each document: "url", "title" and "text",
where "text" is the text to be used for creating questions and answers. 

The pipeline consists of the following steps:
1. Convert the documents to requests for GPT.
2. Make API calls to GPT to generate questions and answers.
3. Filter the generated questions and answers based on scores given by GPT.
4. Correct spelling and grammar in the questions and answers.
5. Add information to the questions and answers.

The pipeline will output the final questions and answers to the file data/queries.jsonl.
The output format is a jsonl file with the following keys for each question and answer pair: "question", "answer",
"question_id", "question_score", "document_score", "url", "title" and "context".
"""

from scripts.create_requests import doc_to_requests
from scripts.call_api import make_API_call
from scripts.filter_queries import filter_queries
from scripts.correct_queries import correct_qas
from scripts.add_info import add_info
from pathlib import Path
import argparse


def main(arguments):

    print("Converting documents to requests...")
    doc_to_requests(arguments.dataset_path, arguments.gpt_model)

    print(f"Making API calls with the {arguments.gpt_model} model...")
    make_API_call(arguments.gpt_model)

    print("Filtering queries...")
    filter_queries()

    if arguments.skip_correction is False:
        print("Adding information to queries...")
        add_info(arguments.dataset_path, Path("data/extra-data/filtered_queries.jsonl"))
        print("Done! Output written to data/queries.jsonl")
    else:
        print("Correcting queries and answers...")
        correct_qas()

        print("Adding information to queries...")
        add_info(
            arguments.dataset_path,
            Path("data/extra-data/corrected_queries.jsonl"),
        )
        print("Done! Output written to data/queries.jsonl")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to the dataset to create questions and answers from.",
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        help="The GPT model to use.",
        default="gpt-4-turbo",
    )
    parser.add_argument(
        "--skip-correction",
        help="Do not correct spelling and grammar in questions and answers.",
        action="store_false",
    )
    args = parser.parse_args()
    main(args)
