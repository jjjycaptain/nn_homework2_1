from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torchvision.datasets.folder import default_loader
import os

class CustomImageFolder(ImageFolder):
    def __init__(self, root, indices, transform):
        super().__init__(root, transform=transform)
        self.samples = [self.samples[i] for i in indices]
        self.targets = [s[1] for s in self.samples]

def get_loaders(
    data_dir: str = "./dataset/caltech-101/",
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 2025,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    torch.manual_seed(seed)

    # image normalization values for ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225], 
    )

    # Adjust to default ImageNet input size, data augmentation
    train_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_test_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # Custom split for local ImageFolder
    full_dataset = ImageFolder(os.path.join(data_dir, "101_ObjectCategories"), loader=default_loader)
    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=full_dataset.targets, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1/3, stratify=[full_dataset.targets[i] for i in temp_idx], random_state=seed
    )

    train_set = CustomImageFolder(os.path.join(data_dir, "101_ObjectCategories"), train_idx, transform=train_tfm)
    val_set = CustomImageFolder(os.path.join(data_dir, "101_ObjectCategories"), val_idx, transform=val_test_tfm)
    test_set = CustomImageFolder(os.path.join(data_dir, "101_ObjectCategories"), test_idx, transform=val_test_tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, val_loader = get_loaders(data_dir="./dataset/caltech-101/", batch_size=32, num_workers=4)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    for images, labels in train_loader:
        print(f"Batch size: {images.size(0)}, Image shape: {images.shape}, Labels: {labels}")
        break  # Just to show one batch