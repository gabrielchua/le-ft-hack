"""
prepare_ft_dataset.py
"""
import json
import pandas as pd

df = pd.read_csv("lmsys-chatbot-arena/train.csv")

# Only keep prompts that are at least 100 chars
df = df[df.prompt.apply(len) > 100]

# Encode Answer
df['answer'] = "A"
df.loc[df.winner_model_b==1, "answer"] = "B"
df.loc[df.winner_tie==1, "answer"] = "TIE"

# Train, Val, Test Split
df_train=df.sample(frac=0.9,random_state=200)
df_val_test=df.drop(df_train.index)
df_val=df_val_test.sample(frac=0.4,random_state=200)
df_test=df_val_test.drop(df_val.index)

def create_jsonl(file_name, df):
    with open(file_name, 'w') as outfile:
        for _, row in df.iterrows():
            example = {
                "messages": [
                    {"role": "system", "content": "Your task is to determine which is the better response to the given prompt - responses A or B. If it is a tie, reply with `TIE`."},
                    {"role": "user", "content": f"Here is the given prompt(s): {row['prompt']} <responses A> {row['response_a']} </responses A> <responses B> {row['response_b']} </responses B>"},
                    {"role": "assistant", "content": row['answer']}
                ]
            }
            json.dump(example, outfile)
            outfile.write('\n')


create_jsonl("arena_train.jsonl", df_train)
create_jsonl("arena_val.jsonl", df_val)
create_jsonl("arena_test.jsonl", df_test)
