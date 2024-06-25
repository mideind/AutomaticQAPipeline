import json
import uuid
from pathlib import Path


def add_info(dataset_path: str, queries_path: str):
    """Add information to queries from the original dataset."""

    original_texts = []
    with open(dataset_path, "r") as f:
        original_data = [json.loads(line) for line in f]
        for orig_text in original_data:
            url = orig_text["url"]
            title = orig_text["title"]
            text = orig_text["text"]
            original_texts.append({"url": url, "title": title, "text": text})

    updated_queries = []
    with open(queries_path, "r") as f:
        queries = [json.loads(line) for line in f]
        for query in queries:
            new_query = {}
            id = query["id"]
            for orig_text in original_texts:
                if id == orig_text["url"]:
                    new_query["question"] = query["question"]
                    new_query["answer"] = query["answer"]
                    new_query["question_id"] = str(uuid.uuid4())
                    new_query["question_score"] = query["question_score"]
                    new_query["document_score"] = query["document_score"]
                    new_query["url"] = query["id"]
                    new_query["title"] = orig_text["title"]
                    new_query["context"] = orig_text["text"]
                    updated_queries.append(new_query)

    if not Path("data").exists():
        Path("data/").mkdir(parents=True, exist_ok=True)
    with open(Path("data/queries.jsonl"), "w") as f:
        for query in updated_queries:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")
