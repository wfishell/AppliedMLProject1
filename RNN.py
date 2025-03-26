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

def train(padded_sequences, sample_count=10):
    """
    Train the model and mark sample_count iterations (selected randomly, after skipping the first 10)
    using NVTX markers. When you run nvprof with --metrics, it will capture kernel metrics (FLOPs, bytes)
    for these marked iterations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    overall_start = time.time()

    # Prepare data
    input_sequences = padded_sequences[:, :-1]
    target_sequences = padded_sequences[:, 1:]
    dataset = TensorDataset(input_sequences, target_sequences)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)

    # Initialize model, loss, optimizer
    ELMo = BiLSTMEmbeddings(vocab_size=10000, embedding_dim=64, lstm_units=2, pad_idx=0).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(ELMo.parameters(), lr=0.01)

    num_epochs = 5
    ELMo.train()

    # Determine total number of iterations and select random ones (skipping the first 10)
    total_batches_per_epoch = len(dataloader)
    total_iterations = num_epochs * total_batches_per_epoch
    # Choose sample_count random global iteration indices from [10, total_iterations)
    sample_indices = set(random.sample(range(10, total_iterations), sample_count))
    print(f"Profiling will occur at global iterations: {sorted(sample_indices)}")

    # List to store timing metrics for the sampled iterations
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
                # Mark this iteration with an NVTX range so nvprof can filter it
                nvtx.range_push(f"Profiled Iteration {global_iter}")
                start_time = time.time()

                outputs = ELMo(batch_inputs)
                outputs = outputs.view(-1, 10000)
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
                outputs = ELMo(batch_inputs)
                outputs = outputs.view(-1, 10000)
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

    return ELMo, profiled_metrics

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
    dataset = load_dataset("ptb_text_only")
    sentence_dict = tokenize_text(dataset, "train")
    token2int, int_sentences = Tokenize_to_Integer(sentence_dict)
    tensor_sentences = [torch.tensor(tokens, dtype=torch.long) for tokens in int_sentences.values()]
    padded_sequences = pad_sequence(tensor_sentences, batch_first=True, padding_value=0)
    print("Shape of padded_sequences:", padded_sequences.shape)

    # Train and randomly mark 10 iterations (not the first 10) for profiling via NVTX markers
    trained_model, metrics = train(padded_sequences, sample_count=10)

    # Optionally, write timing metrics to CSV.
    # write_metrics_to_csv(metrics)

if __name__ == "__main__":
    main()
