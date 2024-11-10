from datasets import load_dataset
import pickle

ds = load_dataset("google-research-datasets/paws","labeled_final")

import torch
from datasets import load_dataset
from sentence_transformers import InputExample, CrossEncoder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Check if GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the PiC/phrase_similarity dataset
dataset = load_dataset("google-research-datasets/paws","labeled_final")
# import pdb;pdb.set_trace()

# Convert the dataset into InputExample format for CrossEncoder
train_samples = [
    InputExample(texts=[item["sentence1"], item["sentence2"]], label=item["label"])
    for item in dataset["train"]
]
val_samples = [
    InputExample(texts=[item["sentence1"], item["sentence2"]], label=item["label"])
    for item in dataset["validation"]
]

# Create DataLoader for the training set
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=128)

# List of models to train
models = [
    "google-bert/bert-large-uncased",
    "FacebookAI/roberta-base",
    "distilbert/distilbert-base-uncased",
    "distilroberta-base",
]

# Initialize lists to store accuracy for plotting
model_names = []
best_accuracies = []
best_epochs = []

# Define a custom evaluation function
def evaluate(model, val_samples):
    val_texts = [sample.texts for sample in val_samples]
    val_labels = [sample.label for sample in val_samples]

    # Predict scores with CrossEncoder
    predictions = model.predict(val_texts)
    predictions = np.argmax(predictions,axis=1)
    predicted_labels = [1 if score >= 0.5 else 0 for score in predictions]  # Convert scores to binary labels

    # Calculate accuracy
    accuracy = accuracy_score(val_labels, predicted_labels)
    return accuracy
num_epochs=5

warmup_steps = int(0.1 * len(train_dataloader) * num_epochs)  # Warm-up steps
learning_rate = 1e-5  # Adjust as needed
model_save_path = "output/crossencoder-pic"
# Loop over the models
all_acc=[]
for model_name in models:
    print(f"Training model: {model_name}")

    # Initialize CrossEncoder with the current model
    model = CrossEncoder(model_name, num_labels=2, device=device)

    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        model.model = torch.nn.DataParallel(model.model)

    # Variables to track the best accuracy and epoch
    best_accuracy = 0
    best_epoch = 0
    accuracies = []

    # Train the model
    for epoch in range(num_epochs):
        model.fit(
            train_dataloader=train_dataloader,
            epochs=1,  # Train for one epoch at a time
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            optimizer_params={"lr": learning_rate}  # Set the learning rate
        )

        # Evaluate after each epoch
        accuracy = evaluate(model, val_samples)
        accuracies.append(accuracy)  # Store accuracy for plotting
        print("Epoch {} Acc {}".format(epoch,accuracy))

        # Save the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1  # Store the epoch (1-indexed)
            # model.save(f"{model_save_path}_{model_name.replace('/', '_')}.bin")  # Save the best model
    all_acc.append(accuracies)

# Save all_acc to a pickle file
with open("all_acc.pkl", "wb") as f:
    pickle.dump(all_acc, f)

#     # Store the results
#     model_names.append(model_name)
#     best_accuracies.append(best_accuracy)
#     best_epochs.append(best_epoch)

#     print(f"Best Model for {model_name} saved with accuracy: {best_accuracy * 100:.2f}% at epoch: {best_epoch}")

# # Plotting the accuracy curves for all models
# plt.figure(figsize=(10, 6))
# for model_name, accuracies in zip(models, accuracies):
#     plt.plot(range(1, num_epochs + 1), accuracies, label=model_name)

# plt.title('Model Accuracy Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid()
# plt.show()

# # Print the best results
# print("\nBest Results:")
# for name, acc, epoch in zip(model_names, best_accuracies, best_epochs):
#     print(f"Model: {name}, Best Accuracy: {acc * 100:.2f}%, Best Epoch: {epoch}")