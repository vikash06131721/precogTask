import requests
import numpy as np
import matplotlib.pyplot as plt
def get_similarity_score(prompt, model="gemma2:2b", url="http://35.208.33.92:5000/api/generate"):
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        # Send the POST request
        response = requests.post(url, json=data)
        
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Return the JSON response (e.g., similarity score)
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None
def prompt_based_similarity(sent1, sent2):
    """
    Template for using language models with prompting for similarity scoring
    Note: This is a demonstration - actual implementation would need an LLM API
    """
    prompt = f"""
    On a scale of 0 to 1, how similar are these sentences semantically and return just the score in float?
    Sentence 1: {sent1}
    Sentence 2: {sent2}
    
    Think step by step:
    1. Compare the main topics/subjects
    2. Compare the actions/verbs
    3. Compare the context and meaning
    4. Consider synonyms and related concepts
    
    Provide a final similarity score between 0 and 1, where:
    0 = completely different meaning
    1 = identical meaning
    """
    # In practice, you would send this prompt to an LLM API
    return prompt
# Example usage
prompt = prompt_based_similarity("one data","a particular statistic")
result = get_similarity_score(prompt)

from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import re

def extract_similarity_score(text):
    match = re.search(r'similarity score is \*\*(\d\.\d+)\*\*', text)
    if match:
        return float(match.group(1))
    return None

ds = load_dataset("PiC/phrase_similarity")
train_sample = ds['train'].shuffle(seed=42).select(range(1000))


import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor



def add_embedding_columns(row):
    # Get embeddings for phrase1 and phrase2
    prompt = prompt_based_similarity(row["phrase1"],row["phrase2"])
    result = get_similarity_score(prompt)

    row["cosine"] = result
    return row


train_sample = train_sample.map(add_embedding_columns)

# for split in ['train', 'validation', 'test']:
#     ds[split] = ds[split].map(
#         lambda x: {
#             'cosine': add_embedding_columns(x["phrase1"],x["phrase2"])
#         }
#     )

cosine_similarities = train_sample["cosine"]
labels = train_sample["label"]
text = [x['response'] for x in cosine_similarities]

df = pd.DataFrame()
df['cos'] = text
df['lbl'] = labels


df.to_csv("promptSim.csv",index=False)
def refined_extract_score(text):
    # Look for numbers that may or may not be surrounded by asterisks and parentheses
    match = re.search(r"(\d\.\d+)", text)  # Find any floating-point number
    return float(match.group(1)) if match else None


df = pd.read_csv("promptSim.csv")
# Apply the extraction function to the 'cos' column
df['cosine'] = df['cos'].apply(refined_extract_score)
df['cosine'] = df['cosine'].fillna(' ')
df = df[df['cosine']!=' ']
df['cosine'] = df['cosine'].apply(lambda x:float(x))
# Display the first few rows to verify the 'cosine' column
cosine_similarities = df['cosine'].tolist()
predictions = [1 if score >= 0.6 else 0 for score in cosine_similarities]
labels = df['lbl'].tolist()

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

plt.title("Lift Plot for Cosine Similarity Predictions For Prompt Llama 1B for Sentence Similarity")
plt.xlabel("Proportion of Predictions (sorted by similarity)")
plt.ylabel("Lift")
plt.legend()
plt.grid()
plt.savefig("llama1bPromptSent.png")
