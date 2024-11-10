from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



ds = load_dataset("PiC/phrase_similarity")


import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def get_embeddings(prompt, model="gemma2:2b", url="http://35.208.33.92:5000/api/embeddings"):
    data = {
        "model": model,
        "prompt": prompt
    }

    try:
        # Make the POST request
        response = requests.post(url, json=data)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Return the JSON response (assumed to contain embeddings)
        emb = response.json()
        return emb['embedding']

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

# Example usage
prompt = "Llamas are members of the camelid family"
embeddings = get_embeddings(prompt)

def add_embedding_columns(row):
    # Get embeddings for phrase1 and phrase2
    row["embedding_phrase1"] = get_embeddings(row["phrase1"])
    row["embedding_phrase2"] = get_embeddings(row["phrase2"])
    return row


ds = ds.map(add_embedding_columns)

for split in ['train', 'validation', 'test']:
    ds[split] = ds[split].map(
        lambda x: {
            'cosine': cosine_similarity(np.array(x['embedding_phrase1']).reshape(1,-1),np.array(x['embedding_phrase2']).reshape(1,-1))[0][0]
        }
    )

cosine_similarities = ds["train"]["cosine"]
labels = ds["train"]["label"]


import numpy as np
import matplotlib.pyplot as plt

# Example data
# cosine_similarities = [...]  # List of cosine similarity scores between 0 and 1
# labels = [...]  # List of corresponding labels (0 or 1)

# Convert cosine similarities to binary predictions (>= 0.5 as 1, < 0.5 as 0)
predictions = [1 if score >= 0.6 else 0 for score in cosine_similarities]

# Sort by predicted similarity scores in descending order
sorted_indices = np.argsort(-np.array(cosine_similarities))
sorted_labels = np.array(labels)[sorted_indices]
sorted_predictions = np.array(predictions)[sorted_indices]

# Calculate cumulative true positives and total positives
cumulative_true_positives = np.cumsum(sorted_labels)
total_positives = sum(labels)

# Calculate lift at each percentile of predictions
lift = cumulative_true_positives / (np.arange(1, len(labels) + 1) * total_positives / len(labels))

# Plot the lift curve
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(labels) + 1) / len(labels), lift, marker='o', label="Lift")
plt.axhline(y=1, color='r', linestyle='--', label="Baseline (Random Guessing)")

plt.title("Lift Plot for Cosine Similarity Predictions For Gemma 2B")
plt.xlabel("Proportion of Predictions (sorted by similarity)")
plt.ylabel("Lift")
plt.legend()
plt.grid()
plt.savefig("gemma2b.png")
