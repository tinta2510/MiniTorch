import re
from minitorch import tensor, Tensor, Module, Parameter
import minitorch
from minitorch.nn import LSTM, GRU, RNN, Linear, Sigmoid, RParam, dropout
from collections import Counter
from itertools import chain

# Load IMDb dataset (using a simplified approach)
# Note: MiniTorch may not have dataset loading utilities like PyTorch
# You would need to implement a custom loader or use a library like datasets
from datasets import load_dataset

dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

# Tokenizer
def simple_tokenizer(text):
    return re.findall(r"\b\w+\b", text.lower())

# Build Vocabulary
counter = Counter(chain(*[simple_tokenizer(x['text']) for x in train_data]))
vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(20000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode(text, max_len=256):
    tokens = simple_tokenizer(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    ids = ids[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(ids))
    return minitorch.tensor(ids)

# Dataset class for MiniTorch
class IMDbDataset:
    def __init__(self, data, max_len=256):
        self.data = data
        self.max_len = max_len
        self.texts = []
        self.labels = []
        
        # Preprocess all data at initialization
        for item in data:
            self.texts.append(encode(item['text'], self.max_len))
            self.labels.append(minitorch.tensor([float(item['label'])]))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# DataLoader equivalent for MiniTorch
class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.indices)
            
        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            texts = []
            labels = []
            
            for idx in batch_indices:
                text, label = self.dataset[idx]
                texts.append(text)
                labels.append(label)
                
            # Stack tensors into batches
            yield minitorch.stack(texts), minitorch.stack(labels)
            
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# Create datasets and dataloaders
train_dataset = IMDbDataset(train_data)
test_dataset = IMDbDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Embedding layer implementation
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = RParam(num_embeddings, embedding_dim)
        
    def forward(self, x):
        # Get embeddings for each token in the batch
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        result = minitorch.zeros((batch_size, seq_len, self.embedding_dim))
        
        for b in range(batch_size):
            for s in range(seq_len):
                idx = int(x[b, s].item())
                if idx != self.padding_idx:
                    result[b, s] = self.weight.value[idx]
        
        return result

# SentimentModel using MiniTorch
class SentimentModel(Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, model_type='GRU'):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, padding_idx=vocab["<PAD>"])
        
        self.model_type = model_type
        if model_type == 'RNN':
            self.rnn = RNN(embed_dim, hidden_dim)
        elif model_type == 'GRU':
            self.rnn = GRU(embed_dim, hidden_dim)
        elif model_type == 'LSTM':
            self.rnn = LSTM(embed_dim, hidden_dim)
            
        self.fc = Linear(hidden_dim, 1)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        
        # Process with RNN/GRU/LSTM
        if self.model_type == 'LSTM':
            output, (hidden, _) = self.rnn(embedded)
        else:
            output = self.rnn(embedded)
            # Get the last output for each sequence
            hidden = output[-1]
            
        # Apply final layer and sigmoid
        return self.sigmoid(self.fc(hidden))

# Binary Cross Entropy Loss
def bce_loss(predictions, targets):
    epsilon = 1e-10  # Small value to avoid log(0)
    predictions = predictions.view(predictions.shape[0])
    targets = targets.view(targets.shape[0])
    
    # Calculate loss: -[y*log(p) + (1-y)*log(1-p)]
    loss = -(targets * (predictions + epsilon).log() + 
             (1 - targets) * (1 - predictions + epsilon).log())
    
    # Return mean loss
    return loss.sum() / loss.shape[0]

# Training and evaluation functions
def train(model, dataloader, optimizer, epochs=1):
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = bce_loss(outputs, labels)
            
            # Backward pass
            model.zero_grad_()
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            predictions = outputs > 0.5
            correct += ((predictions == labels).sum()).item()
            samples += labels.shape[0]
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  [Batch {batch_idx+1}] Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / samples
        print(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}")
        
        # Evaluate
        eval_loss, eval_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch}: Test Loss: {eval_loss:.4f}, Test Acc: {eval_acc:.4f}")
        
    return model

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    samples = 0
    
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = bce_loss(outputs, labels)
        
        # Track statistics
        total_loss += loss.item()
        predictions = outputs > 0.5
        correct += ((predictions == labels).sum()).item()
        samples += labels.shape[0]
        
    return total_loss / len(dataloader), correct / samples

# Optimizer implementation (simplified SGD)
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
        
    def step(self):
        for p in self.parameters:
            if p.value.grad is not None:
                p.update(p.value - self.lr * p.value.grad)
                
    def zero_grad(self):
        for p in self.parameters:
            if p.value.grad is not None:
                p.value.zero_grad_()

# Run experiment with different model types
def run_experiment():
    for model_type in ['RNN', 'GRU', 'LSTM']:
        print(f"\nTraining {model_type} model...")
        model = SentimentModel(len(vocab), model_type=model_type)
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # Train for just 1 epoch for demonstration
        train(model, train_loader, optimizer, epochs=1)

# Run the experiment
run_experiment()