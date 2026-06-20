"""Train SimpleCNN on MNIST and report test accuracy per epoch."""
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloaders
from model import SimpleCNN

EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 128


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    torch.save(model.state_dict(), "simple_cnn.pt")
    print("Saved model weights to simple_cnn.pt")


if __name__ == "__main__":
    main()
