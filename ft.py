"""
ft.py

This script performs the following tasks:
1. Sets up the Mistral client.
2. Uploads training and validation datasets.
3. Starts a training job.
4. Monitors the training job status.
5. Loads the test dataset.
6. Performs asynchronous inference to gather predictions.
7. Evaluates the model's accuracy.
"""

import os
import json
import time
from datetime import datetime

import asyncio
import numpy as np
from mistralai.client import MistralClient
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.models.jobs import WandbIntegrationIn, TrainingParameters

# Setup Mistral Client
api_key = os.environ.get("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)
async_client = MistralAsyncClient(api_key=api_key)

# API key for W&B
wandb_api_key = os.environ.get("WANDB_API_KEY")

# Upload training and validation set
uploaded_files = []

for file in ["train", "val"]:
    file_name = f"data/arena_{file}_v3.jsonl"
    print(file_name)
    with open(file_name, "rb") as f:
        uploaded_files.append(client.files.create(file=(file_name, f)))

file_ids = [file.id for file in uploaded_files]

# Start training job
created_jobs = client.jobs.create(
    model="mistral-small-latest",
    training_files=[file_ids[0]],
    validation_files=[file_ids[1]],
    hyperparameters=TrainingParameters(
        training_steps=50,
        learning_rate=0.0001,
        ),
    integrations=[
            WandbIntegrationIn(
                project="test_api",
                run_name="test",
                api_key=wandb_api_key,
            ).dict()
        ]
)

# Check training status
while True:
    # Retrieve the job details
    retrieved_job = client.jobs.retrieve(created_jobs.id)
    status = retrieved_job.events[0].data["status"]
    # Print the current time
    print(f"Current time: {datetime.now()}, current status is: {status}")
    # Check the job status
    if status == "SUCCESS":
        print("Job succeeded, stopping the loop.")
        break
    # Wait for 60 seconds before the next iteration
    time.sleep(60)

# Load Test Set
with open("arena_test_v2.jsonl", 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Set-up Async Inference
async def predict_preference(session, system_message, prompt, idx):
    """ 
    Use fine-tune model to predict preference.
    
    Args:
        session: The async client session.
        system_message (str): The system message content.
        prompt (str): The user prompt content.
        idx (int): The index of the current prediction.
    
    Returns:
        tuple: The index and the model's response content.
    """
    print(prompt)
    chat_response = await session.chat(
        model=retrieved_job.fine_tuned_model,
        messages=[{"role": "system", "content": system_message},
                  {"role": "user", "content": prompt}]
    )
    return idx, chat_response.choices[0].message['content']

async def gather_predictions(dataset):
    """ 
    Gather predictions for the dataset using async inference.
    
    Args:
        dataset (list): The dataset to predict on.
    
    Returns:
        list: A list of predictions with their indices.
    """
    tasks = [predict_preference(async_client, dataset[i]['messages'][0]['content'], dataset[i]['messages'][1]['content'], i) for i in range(len(dataset))]
    predictions = []
    for i in range(0, len(dataset), 20):  # Process in chunks of 20
        chunk = tasks[i:i+20]
        predictions.extend(await asyncio.gather(*chunk))
    return predictions

# Execute async function to get predictions
predictions_with_indices = asyncio.run(gather_predictions(dataset))

# Ensure the original indexing is maintained and extract predictions
predictions_with_indices.sort(key=lambda x: x[0])
predictions = [pred[1] for pred in predictions_with_indices]

# Sync inference (commented out)
# for i in range(0, N):
#     chat_response = client.chat(
#         model=retrieved_job.fine_tuned_model,
#         messages=[ChatMessage(role='system', content=dataset[i]['messages'][0]['content']),
#                   ChatMessage(role='user', content=dataset[i]['messages'][1]['content'])]
#     )
#     prediction = chat_response.choices[0].message.content
#     ground_truth = dataset[i]['messages'][2]['content']
#     results.append(prediction == ground_truth)
#     print(f"For entry {i}, {ground_truth} {prediction}")

# Evaluation
ground_truth = [dataset[i]['messages'][2]['content'] for i in range(N)]

predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

print(f"Accuracy is: {np.mean(predictions == ground_truth)}")
