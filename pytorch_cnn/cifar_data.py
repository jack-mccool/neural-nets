"""CIFAR-10 data loading via torchvision, with light augmentation on train."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)

TEST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)


def get_dataloaders(batch_size=128, data_dir="./data"):
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=TRAIN_TRANSFORM)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=TEST_TRANSFORM)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
