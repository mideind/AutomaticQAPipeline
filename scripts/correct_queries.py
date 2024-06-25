from transformers import pipeline
from pathlib import Path
import json


def correct(pipeline, input_query, input_answer):
    """Corrects text using the model provided."""

    prefix = "<|iec, margar villur, yfirlesið|> "

    max_length = 512

    # Check whether the input query is too long for the model
    if len(input_query.encode("utf-8")) < max_length:
        corrected_query = (
            pipeline(prefix + input_query.strip(), max_length=512)[0]["generated_text"]
            + "\n"
        )
    else:
        # The input query is too long for the model so we keep the original instead of splitting it up
        corrected_query = input_query

    # Check whether the input answer is too long for the model
    if len(input_answer.encode("utf-8")) < max_length:
        corrected_answer = (
            pipeline(prefix + input_answer.strip(), max_length=512)[0]["generated_text"]
            + "\n"
        )
    else:
        # The input answer is too long for the model so we keep the original instead of splitting it up
        corrected_answer = input_answer

    return corrected_query, corrected_answer


def correct_qas():
    """Corrects text using the model provided."""

    model_dir = "./byt5_M12_clarin"
    gec_pipeline = pipeline(
        "text2text-generation",
        model=model_dir,
        tokenizer="google/byt5-base",
    )

    corrected_queries = []
    with open(Path("data/extra-data/filtered_queries.jsonl"), "r") as f:
        pairs = [json.loads(line) for line in f]
        for pair in pairs:
            query = pair["question"]
            answer = pair["answer"]
            corrected_query, corrected_answer = correct(gec_pipeline, query, answer)
            pair["question"] = corrected_query
            pair["answer"] = corrected_answer
            corrected_queries.append(pair)

    if not Path("data/extra-data").exists():
        Path("data/extra-data/").mkdir(parents=True, exist_ok=True)
    with open(Path("data/extra-data/corrected_queries.jsonl"), "w") as f:
        for pair in corrected_queries:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
