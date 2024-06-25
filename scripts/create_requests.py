import tiktoken
from pathlib import Path
import json
import random

MODEL = "gpt-4-turbo"
SYSTEM_PROMPT = "Þú ert vandvirk aðstoðarmanneskja"
PRE_PROMPT = "Hér er skjal:\n\n"


def count_tokens(text):
    encoding = tiktoken.encoding_for_model(MODEL)
    return len(list(encoding.encode(text)))


def get_json(content, max_tokens):
    """Returns the appropriate JSON object for the given content."""

    temperature = 0.6

    json_obj = {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    return json_obj


def _doc_to_requests(
    namespace_path: Path,
    output_path: Path,
    post_prompt: str,
):
    """Reads a dataset file, creates a request for each document and writes them to a file."""

    max_tokens = 3000
    max_sent_tokens = 110000

    with open(namespace_path, "r") as f:
        namespace = f.readlines()

    with open(output_path, "w") as f:
        json_objects = []
        for i, doc in enumerate(namespace):
            token_count = count_tokens(PRE_PROMPT + doc + post_prompt)
            while token_count > max_sent_tokens:
                doc = json.loads(doc)
                doc["text"] = doc["text"][: len(doc["text"]) // 2]
                doc = json.dumps(doc, ensure_ascii=False)
                token_count = count_tokens(PRE_PROMPT + doc + post_prompt)
            json_obs = get_json(PRE_PROMPT + doc + post_prompt, max_tokens)
            json_objects.append(json_obs)

        random.shuffle(json_objects)

        for object in json_objects:
            f.write(json.dumps(object, ensure_ascii=False) + "\n")


def doc_to_requests(dataset_path: Path):
    """Converts a dataset to requests for GPT-4."""

    output_path = Path("data/extra-data/gpt_requests.jsonl")
    post_prompt_path = Path("prompt.txt")

    with open(post_prompt_path, "r") as file:
        post_prompt = file.read()
    if not Path("data/extra-data").exists():
        Path("data/extra-data/").mkdir(parents=True, exist_ok=True)

    _doc_to_requests(
        namespace_path=dataset_path, output_path=output_path, post_prompt=post_prompt
    )
