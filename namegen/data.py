"""
Prepares and tokenizes a dataset of names for training a character-level language model.
Splits the data into training and validation sets, constructs PyTorch datasets and dataloaders
with padding and custom collation for variable-length sequences.
"""
import os 
import random
from torch.utils.data import DataLoader
from .namedataset import NameDataset

module_dir = os.path.dirname(__file__)

names_path = os.path.abspath(os.path.join(module_dir, '..', 'data', 'names.txt'))

with open(names_path, "r") as read_text:
    names = read_text.read().splitlines()

all_characters = sorted(list(set("".join(names))))


stoi = {ch: i + 1 for (i, ch) in enumerate(all_characters)}
stoi["."] = 0
itos = {i: ch for (ch, i) in stoi.items()}
token_dim = len(all_characters) + 1
PADDING_IDX = token_dim

MAX_SEQ_LEN = max(len(name) for name in names) + 2


random.seed(42)
names_shuffled = names.copy()
random.shuffle(names_shuffled)

n_total = len(names_shuffled)
train_ratio = 0.9  # 90% train, 10% val
n_train = int(train_ratio * n_total)

train_names = names_shuffled[:n_train]
val_names = names_shuffled[n_train:]

train_dataset = NameDataset(train_names, stoi, padding_idx=PADDING_IDX)
val_dataset = NameDataset(val_names, stoi, padding_idx=PADDING_IDX)

batch_size_par = 32
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size_par,
    shuffle=True,
    collate_fn=lambda b: NameDataset.collate_fn(b, stoi, PADDING_IDX),
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size_par,
    shuffle=True,
    collate_fn=lambda b: NameDataset.collate_fn(b, stoi, PADDING_IDX),
)
