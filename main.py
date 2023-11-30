import torch
import numpy as np
import random
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


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
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,   batch_size=5, shuffle=False)


def display_random_images(n = 9, rows = 3, columns = 3):
    plt.style.use('grayscale')
    fig = plt.figure(figsize=(10, 7)) 
    images = []

    for i in range(n):
        id =  random.randint(0,len(train_dataset))
        images.append(torch.Tensor.numpy(train_dataset[id][0][0]))
        fig.add_subplot(rows,columns,i + 1)
        plt.axis('off') 
        plt.title(id)
        plt.imshow(images[i])
    

    #display some images
    plt.show()


display_random_images(12,4,3)