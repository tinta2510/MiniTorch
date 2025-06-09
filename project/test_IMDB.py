from torchtext.datasets import IMDB
from collections import Counter
import matplotlib.pyplot as plt

# Load dataset
train_iter = list(IMDB(split='train'))
test_iter = list(IMDB(split='test'))

# Check number of samples
print(f"Number of training samples: {len(train_iter)}")
print(f"Number of test samples: {len(test_iter)}")

# Label distribution
labels = [label for label, _ in train_iter]
label_counts = Counter(labels)
print(f"Label distribution in training set: {label_counts}")