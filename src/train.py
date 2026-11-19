"""Train a CNN to predict the region (or the object) from a complete drawing.

Same training loop as in class: Adam + cross-entropy, print the training and
validation loss/accuracy every epoch, early stopping on the validation loss
(patience 3), and a plot of the training and validation loss.

Saves the best model (lowest validation loss) to results/cnn_<model>.pt,
the per-epoch numbers to results/history_<model>_<target>.json and the loss
plot to results/loss_<model>_<target>.png.

Run:  python -m src.train --model small
      python -m src.train --model deep
      python -m src.train --model small --target object   # what is drawn
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening a window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from .config import CATEGORIES, RESULTS_DIR, SEED, model_path
from .data import get_loaders
from .model import MODELS
from .regions import REGIONS

# Apple GPU if available, otherwise CPU (in Colab this would be "cuda")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_accuracy_and_loss(model, loader, criterion):
    """Accuracy and average loss of the model on a DataLoader (from lecture)."""
    model.eval()  # turns dropout off
    correct = 0
    total_loss = 0
    with torch.no_grad():  # no gradients needed when we're only evaluating
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # the class with the highest score
            correct += (pred == target).sum().item()
            # criterion gives the batch's average loss; multiply by the batch size
            # so dividing by the dataset size below gives the true average
            total_loss += criterion(output, target).item() * len(data)
    return correct / len(loader.dataset), total_loss / len(loader.dataset)


def plot_losses(train_losses, val_losses, path):
    """Training vs. validation loss per epoch; the gap between them shows overfitting."""
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path)
    plt.close()


def main():
    # Command-line options, e.g. python -m src.train --model small --target object
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="deep")
    parser.add_argument("--target", choices=["region", "object"], default="region")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    torch.manual_seed(SEED)  # same starting weights every run
    train_loader, val_loader, _, mean, std = get_loaders(args.target)
    n_classes = len(REGIONS) if args.target == "region" else len(CATEGORIES)

    model = MODELS[args.model](n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"model={args.model}  target={args.target}  device={device}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_loss = float("inf")
    wait = 0  # epochs in a row without a better validation loss

    for epoch in range(args.epochs):
        model.train()  # turns dropout on
        train_correct = 0
        train_loss_total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()              # clear the gradients from the last batch
            output = model(data)               # forward pass: scores for each class
            loss = criterion(output, target)
            loss.backward()                    # backpropagation: compute gradients
            optimizer.step()                   # update the weights

            # Reuse the same forward pass to track training accuracy and loss
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_loss_total += loss.item() * len(data)

        train_accuracy = train_correct / len(train_loader.dataset)
        train_loss = train_loss_total / len(train_loader.dataset)
        val_accuracy, val_loss = get_accuracy_and_loss(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print("Epoch:", epoch + 1)
        print("Training loss:", round(train_loss, 4), " Training accuracy:", round(train_accuracy * 100, 2), "%")
        print("Validation loss:", round(val_loss, 4), " Validation accuracy:", round(val_accuracy * 100, 2), "%",
              flush=True)

        # Early stopping: save the model whenever validation loss improves,
        # and stop after 3 epochs in a row without improvement
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            # mean and std are saved too, so the app can normalize new drawings the same way
            torch.save({"model": model.state_dict(), "mean": mean, "std": std},
                       model_path(args.model, args.target))
        else:
            wait += 1

        if wait == 3:
            print("Early stopping")
            break

    print("Highest validation accuracy:", round(max(val_accuracies) * 100, 2), "%")

    # Save the per-epoch numbers and the loss plot to results/
    name = f"{args.model}_{args.target}"
    history = [{"epoch": i + 1, "train_loss": tl, "val_loss": vl, "val_acc": va}
               for i, (tl, vl, va) in enumerate(zip(train_losses, val_losses, val_accuracies))]
    (RESULTS_DIR / f"history_{name}.json").write_text(json.dumps(history, indent=2))
    plot_losses(train_losses, val_losses, RESULTS_DIR / f"loss_{name}.png")


if __name__ == "__main__":
    main()
