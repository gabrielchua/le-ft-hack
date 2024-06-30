"""
ft.py
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
from mistralai.models.jobs import TrainingParameters

# Setup Mistral Client
api_key = os.environ.get("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)

# Upload training and validation set
uploaded_files = []

for file in ["train", "val"]:
    file_name = f"data/arena_{file}.jsonl"
    print(f"Uploading {file_name}")
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
        )
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

# Test FT-ed model
chat_response = client.chat(
    model=retrieved_job.fine_tuned_model,
    messages=[ChatMessage(role='user', content='What is the best French cheese?')]
)

# Load Test Set
with open("arena_test_v2.jsonl", 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Evaluate model
N = len(dataset)
results = []

# Set-up Async Inference
async def predict_preference(session, system_message, prompt):
    """ Use fine-tune model to predict preference """
    print(prompt)
    chat_response = await session.chat(
        model=retrieved_job.fine_tuned_model,
        messages=[ChatMessage(role='system', content=system_message),
                  ChatMessage(role='user', content=prompt)]
    )
    return chat_response.choices[0].message.content

async def gather_predictions(number_predictions):
    """ Gather Predictions """
    tasks = [predict_preference(async_client, dataset[i]['messages'][0]['content'], dataset[i]['messages'][1]['content']) for i in range(number_predictions)]
    predictions = []
    for i in range(0, number_predictions, 20):  # Process in chunks of 20
        chunk = tasks[i:i+20]
        predictions.extend(await asyncio.gather(*chunk))
    return predictions

predictions = asyncio.run(gather_predictions(N))

# Sync inference
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

print(f"Accuracy is: {np.mean(predictions==ground_truth)}")
