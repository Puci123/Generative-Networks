import torch
import torchvision
from torchvision import datasets, transforms


# Przygotowanie transformacji danych
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Ładowanie danych
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Tworzenie obiektów DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=5, shuffle=False)
