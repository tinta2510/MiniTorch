import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
from itertools import chain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IMDb dataset
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
    return torch.tensor(ids)

# 4. Dataset Class
class IMDbDataset(Dataset):
    def __init__(self, data, max_len=256):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = int(self.data[idx]['label'])
        return encode(text, self.max_len), torch.tensor(label, dtype=torch.float)

train_dataset = IMDbDataset(train_data)
test_dataset = IMDbDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 5. Model Definition
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, model_type='GRU'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<PAD>"])
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[model_type]
        self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.model_type = model_type

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        if self.model_type == 'LSTM':
            hidden = hidden[0]  # LSTM returns (hidden, cell)
        return torch.sigmoid(self.fc(hidden[-1]))

# 6. Training and Evaluation
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, correct, total_samples = 0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += ((outputs > 0.5) == labels).sum().item()
        total_samples += labels.size(0)

        # Simple print every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  [Batch {batch_idx+1}] Loss: {loss.item():.4f}")
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += ((outputs > 0.5) == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

# 7. Run Experiment
def run(model_type='GRU', epochs=1):
    model = SentimentModel(len(vocab), model_type=model_type).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"[{model_type}] Epoch {epoch}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# 8. Try All Three Models
for model_type in ['RNN', 'LSTM', 'GRU']:
    run(model_type)
