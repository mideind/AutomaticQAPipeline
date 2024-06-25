from pathlib import Path
import json


def filter_queries():

    min_doc_score: float = 0.7
    min_question_score: float = 0.7
    queries_path = Path("data/extra-data/gpt_responses.jsonl")
    output_path = Path("data/extra-data/filtered_queries.jsonl")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(queries_path, "r") as f:
        data = []
        doc_scores = []
        question_scores = []
        bad_doc_ids = []
        for line in f.readlines():
            jline = json.loads(line)
            doc_scores.append(jline["document_score"])
            question_scores.append(jline["question_score"])
            if jline["question"] == "" or jline["document_score"] < min_doc_score:
                bad_doc_ids.append(jline["id"])
                continue
            if jline["question_score"] < min_question_score:
                continue
            data.append(jline)

    with open(output_path, "w") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
