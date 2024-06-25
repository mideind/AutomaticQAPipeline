from openai import OpenAI
import json
from pathlib import Path

client = OpenAI()


def make_API_call():
    """Makes an API call to the OpenAI API and writes the responses to a file."""

    with open("./data/extra-data/gpt_requests.jsonl", "r") as f:
        requests = [json.loads(line) for line in f]

        json_objs = []
        for request in requests:

            response = client.chat.completions.create(
                model="gpt-4-turbo", messages=request["messages"]
            )

            output = response.choices[0].message.content
            try:
                json_output = json.loads(output)
            except json.JSONDecodeError:
                print(
                    "JSON file could not be decoded. Do you want to manually correct it or skip it? (C/s)"
                )
                print(output)
                choice = input()
                if choice == "C":
                    print("Please enter the corrected JSON object:\n")
                    json_output = json.loads(input())
                else:
                    continue

            json_objs.append(json_output)

        if not Path("data/extra-data").exists():
            Path("data/extra-data/").mkdir(parents=True, exist_ok=True)
        with open(Path("data/extra-data/gpt_responses.jsonl"), "w") as f:
            for json_obj in json_objs:
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
