import math
import random
import time
import torch
import torch.cuda.nvtx as nvtx  # for NVTX markers
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import csv  # for writing to CSV
from torchinfo import summary  # For the layer-by-layer summary

# Positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on 
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# A simple transformer encoder model for token embeddings.
class SimpleTransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, feedforward_dim, num_encoder_layers, pad_idx, dropout=0.1):
        super(SimpleTransformerEmbeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        # Create one or more transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoder(embedded)
        # Transformer expects input of shape (seq_len, batch_size, embedding_dim)
        embedded = embedded.transpose(0, 1)
        transformer_out = self.transformer_encoder(embedded)
        # Transpose back: (batch_size, seq_len, embedding_dim)
        transformer_out = transformer_out.transpose(0, 1)
        logits = self.fc(transformer_out)
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
    # Reserve 0 for the padding index
    token2int = {token: idx + 1 for idx, token in enumerate(sorted(vocab))}
    int_sentences = {
        idx: [token2int[token] for token in tokens]
        for idx, tokens in sentence_dict.items()
    }
    return token2int, int_sentences

def train(padded_sequences, sample_count=10):
    """
    Train the transformer model and mark sample_count iterations (selected randomly, after skipping the first 10)
    using NVTX markers for profiling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    overall_start = time.time()

    # Prepare data: input is all tokens except the last; target is all tokens except the first.
    input_sequences = padded_sequences[:, :-1]
    target_sequences = padded_sequences[:, 1:]
    dataset = TensorDataset(input_sequences, target_sequences)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)

    # Initialize the transformer model.
    vocab_size = 10000  # or len(token2int) + 1 to match your dataset exactly
    embedding_dim = 64
    num_heads = 4
    feedforward_dim = 128
    num_encoder_layers = 1  # you can increase this if desired
    pad_idx = 0

    transformer_model = SimpleTransformerEmbeddings(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        num_encoder_layers=num_encoder_layers,
        pad_idx=pad_idx,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(transformer_model.parameters(), lr=0.01)

    num_epochs = 5
    transformer_model.train()

    total_batches_per_epoch = len(dataloader)
    total_iterations = num_epochs * total_batches_per_epoch
    # Select random iterations (skipping the first 10) to mark with NVTX for profiling.
    sample_indices = set(random.sample(range(10, total_iterations), sample_count))
    print(f"Profiling will occur at global iterations: {sorted(sample_indices)}")

    profiled_metrics = []
    global_iter = 0

    for epoch in range(num_epochs):
        print(f"--- Starting Epoch {epoch+1}/{num_epochs} ---")
        total_loss = 0.0

        for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader, start=1):
            global_iter += 1
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()

            if global_iter in sample_indices:
                nvtx.range_push(f"Profiled Iteration {global_iter}")
                start_time = time.time()

                outputs = transformer_model(batch_inputs)
                outputs = outputs.view(-1, vocab_size)
                batch_targets_flat = batch_targets.view(-1)
                loss = criterion(outputs, batch_targets_flat)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()

                end_time = time.time()
                nvtx.range_pop()

                iteration_time = end_time - start_time
                profiled_metrics.append({
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'global_iteration': global_iter,
                    'loss': loss.item(),
                    'iteration_time': iteration_time
                })

                print(f"Epoch {epoch+1}, Batch {batch_idx}, Global Iter {global_iter}, Loss: {loss.item():.4f} [Profiled], Time: {iteration_time:.4f}s")
            else:
                outputs = transformer_model(batch_inputs)
                outputs = outputs.view(-1, vocab_size)
                batch_targets_flat = batch_targets.view(-1)
                loss = criterion(outputs, batch_targets_flat)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Global Iter {global_iter}, Loss: {loss.item():.4f}")

            total_loss += loss.item()

        avg_loss = total_loss / total_batches_per_epoch
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}\n")

    overall_end = time.time()
    elapsed_training_time = overall_end - overall_start
    print(f"Total Training Time (seconds): {elapsed_training_time:.2f}")

    if profiled_metrics:
        avg_time = sum(m['iteration_time'] for m in profiled_metrics) / len(profiled_metrics)
        print(f"Average Profiled Iteration Time: {avg_time:.4f}s")
    else:
        print("No iterations were profiled.")

    return transformer_model, profiled_metrics

def write_metrics_to_csv(metrics, filename='profiled_metrics.csv'):
    """
    Write a list of dictionaries (metrics) to a CSV file.
    Each dictionary is expected to have the same keys.
    """
    if metrics:
        keys = metrics[0].keys()
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(metrics)
        print(f"Metrics written to {filename}")
    else:
        print("No metrics to write.")

def main():
    # Load and prepare data
    dataset = load_dataset("ptb_text_only")
    sentence_dict = tokenize_text(dataset, "train")
    token2int, int_sentences = Tokenize_to_Integer(sentence_dict)
    tensor_sentences = [torch.tensor(tokens, dtype=torch.long) for tokens in int_sentences.values()]
    padded_sequences = pad_sequence(tensor_sentences, batch_first=True, padding_value=0)
    print("Shape of padded_sequences:", padded_sequences.shape)

    # Train the transformer model and randomly mark iterations for profiling.
    trained_model, metrics = train(padded_sequences, sample_count=10)

    # Optionally, write timing metrics to CSV.
    # write_metrics_to_csv(metrics)

    # Move the model to CPU so that the summary's dummy input is also on CPU.
    trained_model.cpu()

    # Generate a layer-by-layer summary using a dummy input.
    # The dummy input's shape is (sequence_length,) where sequence_length equals padded_sequences.shape[1]-1.
    model_summary = summary(
        trained_model,
        input_size=(padded_sequences.shape[1] - 1,),
        dtypes=[torch.long],
        col_names=["input_size", "output_size", "num_params", "params_percent"],
        verbose=0  # Suppress console printing
    )

    # Write the model summary to a text file.
    with open("model_summary.txt", "w") as f:
        f.write(str(model_summary))

    print("Model summary has been written to model_summary.txt")

if __name__ == "__main__":
    main()
