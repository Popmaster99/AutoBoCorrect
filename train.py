import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from spell_core import (
    MODEL_PATH,
    SpellDataset,
    SpellSeq2Seq,
    build_collate_fn,
    build_lookup_tables,
    build_vocab,
    load_jsonl_dataset,
    split_dataset,
)


def compute_accuracy(model, loader, pad_idx):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            decoder_inputs = batch_y[:, :-1]
            targets = batch_y[:, 1:]

            logits = model(batch_x, decoder_inputs)
            predictions = logits.argmax(dim=-1)

            mask = targets != pad_idx
            correct += (predictions[mask] == targets[mask]).sum().item()
            total += mask.sum().item()

    return correct / max(total, 1)


def run_epoch(
    model: SpellSeq2Seq,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer | None = None,
    pad_idx: int = 0,
) -> tuple[float, float]:
    """Run one epoch and return loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        decoder_inputs = batch_y[:, :-1]
        targets = batch_y[:, 1:]

        if training:
            optimizer.zero_grad()

        logits = model(batch_x, decoder_inputs)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        predictions = logits.argmax(dim=-1)
        mask = targets != pad_idx
        correct += (predictions[mask] == targets[mask]).sum().item()
        total += mask.sum().item()

    accuracy = correct / max(total, 1)
    return total_loss / max(len(loader), 1), accuracy


def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path: Path = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def train_model(
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    validation_fraction: float = 0.1,
    patience: int = 6,
) -> None:
    data = load_jsonl_dataset()
    print(f"Loaded: {len(data)}")

    training_data, validation_data = split_dataset(data, validation_fraction=validation_fraction)
    print(f"Train rows: {len(training_data)} | Validation rows: {len(validation_data)}")

    char2idx, idx2char = build_vocab(data)
    pad_idx = char2idx["<PAD>"]

    train_dataset = SpellDataset(training_data, char2idx)
    val_dataset = SpellDataset(validation_data, char2idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(pad_idx),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(pad_idx),
    )

    model = SpellSeq2Seq(
        vocab_size=len(char2idx),
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_state = None
    best_val_loss = float("inf")
    bad_epochs = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, pad_idx)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, None, pad_idx)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    lookup_tables = build_lookup_tables(data)
    torch.save(
        {
            "model": model.state_dict(),
            "char2idx": char2idx,
            "idx2char": idx2char,
            "config": {
                "embedding_dim": 128,
                "hidden_size": 256,
                "num_layers": 2,
                "dropout": 0.2,
            },
            "lookup_tables": {
                "exact_corrections": lookup_tables.exact_corrections,
                "noisy_forms": lookup_tables.noisy_forms,
                "targets": sorted(lookup_tables.targets),
            },
        },
        MODEL_PATH,
    )
    print(f"Model saved to {MODEL_PATH}")

    plot_metrics(
        train_losses, val_losses, train_accs, val_accs,
        save_path=Path(__file__).parent / "training_metrics.png"
    )


if __name__ == "__main__":
    train_model()
