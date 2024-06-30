"""

"""
import json
import time
import os
from datetime import datetime

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.models.jobs import TrainingParameters

# Setup Mistral Client
api_key = os.environ.get("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)

# Upload training and validation set
uploaded_files = []

for file in ["train", "val"]:
    file_name = f"arena_{file}.jsonl"
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
with open("arena_test.jsonl", 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Evaluate model
N = len(dataset)

results = []

for i in range(0, N):
    chat_response = client.chat(
        model=retrieved_job.fine_tuned_model,
        messages=[ChatMessage(role='system', content=dataset[i]['messages'][0]['content']),
                  ChatMessage(role='user', content=dataset[i]['messages'][1]['content'])]
    )
    prediction = chat_response.choices[0].message.content
    ground_truth = dataset[i]['messages'][2]['content']
    results.append(prediction == ground_truth)
    print(f"For entry {i}, {ground_truth} {prediction}")
