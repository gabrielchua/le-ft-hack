"""
prepare_ft_dataset.py
"""
import ast
import itertools
import json
import pandas as pd

# Load the dataset
df = pd.read_csv("data/lmsys-chatbot-arena/train.csv")

# Create a copy of the dataframe to swap columns and avoid positional bias
df_swap = df.copy()

# Swap the columns to avoid minimise bias
df_swap['response_a'] = df['response_b']
df_swap['response_b'] = df['response_a']
df_swap['winner_model_a'] = df['winner_model_b']
df_swap['winner_model_b'] = df['winner_model_a']

# Concatenate the original and swapped dataframes
df = pd.concat([df, df_swap], axis=0)
df = df.reset_index(drop=True)

# Encode the answer based on the winner model
df['answer'] = "A"
df.loc[df.winner_model_b == 1, "answer"] = "B"
df.loc[df.winner_tie == 1, "answer"] = "TIE"

def safe_eval(x):
    """
    Safely convert a string representation of a list to an actual list.
    """
    try:
        return ast.literal_eval(x)
    except:
        return []

# Convert list-like strings to actual lists
for col in ["prompt", "response_a", "response_b"]:
    df[col] = df[col].apply(safe_eval)

def recreate_conversation(list1, list2, identity):
    """
    Recreate a conversation by interleaving two lists of messages.
    
    Args:
        list1 (list): List of user messages.
        list2 (list): List of assistant messages.
        identity (str): Identity of the assistant (e.g., "A" or "B").
    
    Returns:
        str: Interleaved conversation as a single string.
    """
    interleaved = itertools.zip_longest(list1, list2, fillvalue='')
    script_parts = []
    for a, b in interleaved:
        if a:
            script_parts.append(f"User: {a}")
        if b:
            script_parts.append(f"Assistant {identity}: {b}")
    return "; ".join(script_parts)

# Convert the prompts and responses to conversation format
df["A"] = df.apply(lambda row: recreate_conversation(row['prompt'], row['response_a'], "A"), axis=1)
df["B"] = df.apply(lambda row: recreate_conversation(row['prompt'], row['response_b'], "B"), axis=1)

# Split the data into training, validation, and test sets
df_train = df.sample(frac=0.9, random_state=200)
df_val_test = df.drop(df_train.index)
df_val = df_val_test.sample(frac=0.4, random_state=200)
df_test = df_val_test.drop(df_val.index)

def create_jsonl_file(file_name, df):
    """
    Create a JSONL file from a dataframe.
    
    Args:
        file_name (str): The name of the output file.
        df (pd.DataFrame): The dataframe to convert to JSONL.
    """
    with open(file_name, 'w') as outfile:
        for _, row in df.iterrows():
            example = {
                "messages": [
                    {"role": "system", "content": "Your task is to determine which conversation <A> or <B> responded better to the user. If it is a tie, reply with `TIE`."},
                    {"role": "user", "content": f"<conversation A> {row['A']} </conversation A> <conversation B> {row['B']} </conversation B>"},
                    {"role": "assistant", "content": row['answer']}
                ]
            }
            json.dump(example, outfile)
            outfile.write('\n')

# Create JSONL files for training, validation, and test sets
create_jsonl_file("data/arena_train_v3.jsonl", df_train)
create_jsonl_file("data/arena_val_v3.jsonl", df_val)
create_jsonl_file("data/arena_test_v3.jsonl", df_test)
