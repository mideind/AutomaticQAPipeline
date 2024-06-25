# Automatic QA Pipeline

A pipeline to automatically generate questions and answers, which pertain to Icelandic culture and/or history, from a dataset.

To run the script you will need a python3 environment, version 3.10.13 or older. Install the required dependencies by running
> pip install -r requirements.txt

The input dataset is assumed to be a jsonl file with the following keys for each document: "url", "title" and "text", where "text" is the text to be used for creating questions and answers. An example of such a dataset is the [Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia). 

The pipeline consists of the following steps:
1. Convert the documents to requests for a GPT model.
2. Make API calls to a GPT model to generate questions and answers.
3. Filter the generated questions and answers based on scores given by GPT.
4. Correct spelling and grammar in the questions and answers.
5. Add information to the questions and answers.

The pipeline can be run from the command line by running the following command:
> python run_qa_pipe.py --dataset-path path/to/dataset

The default GPT model which is used is 'gpt-4-turbo', but this can be changed by including the `gpt-model` flag with the relevant GPT model's name.
> python run_qa_pipe.py --dataset-path path/to/dataset --gpt-model name-of-gpt-model

In order to make an API call to GPT, you need an API key. Instructions on how to obtain such a key are in the [OpenAI API reference](https://platform.openai.com/docs/api-reference/authentication).

The pipeline defaults to correcting spelling and grammar in the questions and answers, but it can be skipped by including the `skip-correction` flag:
> python run_qa_pipe.py --dataset-path path/to/dataset --skip-correction

In order to correct spelling and grammar, the [Byte-Level Neural Error Correction Model for Icelandic](http://hdl.handle.net/20.500.12537/324) must be downloaded and placed within this repository.

The pipeline will output the final questions and answers to the file `data/queries.jsonl`. The output format is a jsonl file with the following format for each question and answer pair: "question", "answer", "question_id", "question_score", "document_score", "url", "title" and "context". Outputs from each step in the pipeline are written to `data/extra-data/`.