"""MNIST data loading via torchvision, downloaded into ./data on first run."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ]
)


def get_dataloaders(batch_size=128, data_dir="./data"):
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=TRANSFORM)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=TRANSFORM)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
