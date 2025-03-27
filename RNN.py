import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import csv
from torchinfo import summary  # For the layer-by-layer summary

class BiLSTMEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, pad_idx):
        super(BiLSTMEmbeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_units * 2, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

def tokenize_text(dataset, split_name):
    sentence_dict = {}
    for idx, example in enumerate(dataset[split_name]):
        tokens = example["sentence"].split()
        sentence_dict[idx] = tokens
    return sentence_dict

def Tokenize_to_Integer(sentence_dict):
    vocab = set()
    for tokens in sentence_dict.values():
        vocab.update(tokens)
    token2int = {token: idx + 1 for idx, token in enumerate(sorted(vocab))}
    int_sentences = {
        idx: [token2int[token] for token in tokens]
        for idx, tokens in sentence_dict.items()
    }
    return token2int, int_sentences

def train(padded_sequences):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data: shift input and target by one token.
    input_sequences = padded_sequences[:, :-1]
    target_sequences = padded_sequences[:, 1:]
    dataset = TensorDataset(input_sequences, target_sequences)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)

    # Initialize model, loss, optimizer
    model = BiLSTMEmbeddings(vocab_size=10000, embedding_dim=64, lstm_units=2, pad_idx=0).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 5
    model.train()

    for epoch in range(num_epochs):
        print(f"--- Starting Epoch {epoch+1}/{num_epochs} ---")
        total_loss = 0.0

        for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader, start=1):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()

            outputs = model(batch_inputs)
            outputs = outputs.view(-1, 10000)
            batch_targets_flat = batch_targets.view(-1)
            loss = criterion(outputs, batch_targets_flat)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}\n")

    return model

def main():
    # Load and prepare data
    dataset = load_dataset("ptb_text_only")
    sentence_dict = tokenize_text(dataset, "train")
    token2int, int_sentences = Tokenize_to_Integer(sentence_dict)
    tensor_sentences = [torch.tensor(tokens, dtype=torch.long) for tokens in int_sentences.values()]
    padded_sequences = pad_sequence(tensor_sentences, batch_first=True, padding_value=0)
    print("Shape of padded_sequences:", padded_sequences.shape)

    # Train the model
    trained_model = train(padded_sequences)

    # Move model to CPU so the summary's dummy input is also on CPU
    trained_model.cpu()

    # Generate a layer-by-layer summary string
    # Use dtypes=[torch.long] so that the dummy input matches the embedding's integer indices.
    model_summary = summary(
        trained_model,
        input_size=(padded_sequences.shape[1] - 1,),
        dtypes=[torch.long],
        col_names=["input_size", "output_size", "num_params", "params_percent"],
        verbose=0  # Suppress console printing
    )

    # Write the summary to a text file
    with open("model_summary.txt", "w") as f:
        f.write(str(model_summary))

    print("Model summary has been written to model_summary.txt")

if __name__ == "__main__":
    main()
