# le finetuning hackathon
Submission for [Mistral Finetuning Hackathon - June 2024](https://mistral.ai/news/2024-ft-hackathon/)

## Summary

We fine-tune `mistral-small-latest` to predict which LLM response will be preferred by a human, using data from the lmsys chatbot arena.

The dataset contains one prompt, responses from two different LLMs, and which of the LLM gave the preferred answer or whether it was a tie.

With 100K+ training examples, 50 training steps and a learning rate of 10^-4, we achieve an accuracy of ~42%. This is 9% point higher than a naive baseline of randomly guessing.

## Set-up

1. Clone this repo

2. Install the dependencies with the following:

```bash
pip install -r requirements.txt
```

3. Download the data from [Kaggle](https://www.kaggle.com/competitions/lmsys-chatbot-arena/data) and add it to `data` folder

4. Run the following:

```bash
python prepare_ft_dataset.py
python ft.py
```

`prepare_ft_dataset.py` will convert the Kaggle dataset into a `.jsonl` file for fine-tuning, and `ft.py` will upload the files to Mistral's platform, initialise the fine-tuning job, and evaluate the fine-tuned model.
