import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define a simple RNN model that learns word embeddings
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleRNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embed = self.embeddings(x)  # [batch_size, seq_length, embedding_dim]
        rnn_out, _ = self.rnn(embed)  # [batch_size, seq_length, hidden_dim]
        logits = self.linear(rnn_out)  # [batch_size, seq_length, vocab_size]
        return logits

def dataprep():
    # Load the Penn Treebank training split and create a corpus string.
    train_iter = PennTreebank(split='train')
    corpus = " ".join(list(train_iter))
    
    # Tokenize the corpus.
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(corpus)
    
    # Build vocabulary from tokens, adding a special token for unknown words.
    vocab = build_vocab_from_iterator([tokens], specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    # Convert tokens to indices.
    indices = [vocab[token] for token in tokens]
    return indices, vocab, corpus

def main():
    # Data preparation.
    indices, vocab, corpus = dataprep()
    # Hyperparameters.
    vocab_size = len(vocab)
    print(vocab_size)
    embedding_dim = 128
    hidden_dim = 256
    sequence_length = 30  # number of tokens per sequence.
    batch_size = 100
    num_epochs = 2  # for demonstration; increase for real training.

    # Instantiate the model, loss function, and optimizer.
    model = SimpleRNN(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy training data by splitting the indices into batches.
    num_batches = len(indices) // (sequence_length * batch_size)
    trimmed_len = num_batches * sequence_length * batch_size
    inputs_all = torch.tensor(indices[:trimmed_len])
    inputs_all = inputs_all.view(batch_size, -1)  # shape: [batch_size, num_batches * sequence_length]
    
    # Measure training time.
    start_time = time.time()
    
    # Training loop.
    for epoch in range(num_epochs):
        print(f'Began Training {epoch}')
        epoch_loss = 0.0
        # Loop over each batch within the epoch.
        for i in range(0, inputs_all.size(1) - sequence_length, sequence_length):
            print(f'Iterating over batch')
            x = inputs_all[:, i:i+sequence_length]
            target = inputs_all[:, i+1:i+sequence_length+1]
            
            optimizer.zero_grad()
            outputs = model(x)
            # Flatten the output and target tensors for loss computation.
            loss = criterion(outputs.view(-1, vocab_size), target.reshape(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
