# LLM OCEAN (Big Five) Personality Evaluation

## Overview

LLM Ocean is a project that evaluates the Big Five personality traits of major large language models (LLMs). The aim is to assess base personality profiles using a standardized test via API and share the results publicly. This sets a foundation for understanding LLM behavioral tendencies, with potential for later expansion (e.g., testing trait flexibility).

## Value Proposition

...

## Run the project

Setup virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
poetry install
```

Setup .env file:
```bash
cp .env.example .env
```

Run all evaluations:
```bash
python eval.py
```

Run a single evaluation:
```bash
python eval.py --model openai:gpt-4o-mini
```

Setup precommit:
```bash
pre-commit install
```

## Todo

- [ ] Better readme
- [ ] Add an Ollama model
- [ ] Add configs for temperature
- [ ] Add eval and dataset to huggingface


## Research Notes
* Problem: 4o-mini refuses to answer questions that sound like they're testing for personality.
  * Example: 
    * Input: 'I see myself as someone who is talkative.'
    * Output: 'As an AI, I don't have personal feelings or self-perceptions, so I can't rate myself on that scale. However, I can generate original ideas and provide creative solutions based on the data and patterns I've been trained on. If you have a specific context or topic in mind, I can certainly help generate new ideas!'
  * Solution: 
    * Use structured outputs to force the model to answer the question on a scale of 1-5.