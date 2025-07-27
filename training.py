"""Trains the transformer model for character-level name generation using both token and position inputs."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from namegen.config import *
from namegen.data import (MAX_SEQ_LEN, PADDING_IDX, token_dim,
                          train_dataloader, val_dataloader)
from namegen.model import Transformermodel

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA (GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = Transformermodel(
    token_dim=token_dim,
    model_dim=model_dim,
    max_seq_length=MAX_SEQ_LEN,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    padding_idx=PADDING_IDX,
)
model = model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PADDING_IDX)
optimizer = optim.Adam(model.parameters(), lr=LR)

start_epoch = 0
checkpoint_path = "checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"Resumed from checkpoint at epoch {start_epoch}")
wandb.init(project="Name_Decoderonly_Transformer")

for epoch in range(start_epoch, num_epochs):
    print("New epoch")
    model.train()
    running_train_loss = 0.0

    for token_batch, pos_batch in train_dataloader:
        token_batch = token_batch.to(device)
        pos_batch = pos_batch.to(device)
        input_token = token_batch[:, :-1]
        input_pos = pos_batch[:, :-1]
        target_token = token_batch[:, 1:]

        logits = model(input_token, input_pos)
        B, T, V = logits.shape

        loss = criterion(logits.reshape(B * T, V), target_token.reshape(B * T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_training_loss = running_train_loss / len(train_dataloader)
    print(f"Epoch: {epoch + 1}, Training Loss: {avg_training_loss:.4f}")
    wandb.log({"Training Loss": avg_training_loss, "epoch": epoch + 1})

    if (epoch + 1) % 10 == 0:
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at epoch {epoch + 1}")

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for token_batch, pos_batch in val_dataloader:
            token_batch = token_batch.to(device)
            pos_batch = pos_batch.to(device)
            input_token = token_batch[:, :-1]
            input_pos = pos_batch[:, :-1]
            target_token = token_batch[:, 1:]

            logits = model(input_token, input_pos)
            B, T, V = logits.shape
            val_loss = criterion(logits.reshape(B * T, V), target_token.reshape(B * T))
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_dataloader)
    print(f"Epoch: {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")
