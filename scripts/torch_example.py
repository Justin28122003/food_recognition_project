import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader
import random 
import typing

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

# 1. Data Generation
num_samples = 1000
# Data of class A (centered at 0)
A = np.random.normal(0, 0.75, size=(num_samples, 2))
# Data of class B (centered at 3)
B = np.random.normal(3, 0.75, size=(num_samples, 2))

# Labels: A is 0, B is 1
y = [0 if i < num_samples else 1 for i in range(num_samples * 2)]
input_data = np.concatenate([A, B], axis=0)

# Visualize the dataset
plt.figure(figsize=(10, 7))
plt.scatter(input_data[:, 0], input_data[:, 1], c=y, alpha=0.5)
plt.title("Generated Data Distribution")
plt.grid()
plt.savefig("data_distribution.png")
plt.close() # Close plot to free up memory

class Model(nn.Module):
    def __init__(self, input_dim: int=2, hidden_dim: int=5, output: int=1):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, input_dim=2] 
        output = self.fc1(x) # shape: [batch_size, hidden_dim=5]
        output = self.relu(output)
        output = self.fc2(output)
        sigmoid = self.sigmoid(output)
        return sigmoid # shape: [batch_size, 1]

class ExampleData(Dataset):
    def __init__(self, datapoints: np.array, labels: np.array):
        self.datapoints = torch.tensor(datapoints)
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.datapoints) # datapoints.shape[0]

    def __getitem__(self, index: int):
        x = self.datapoints[index].float()
        y = self.labels[index].float()
        return x, y

#model = Model(hidden_dim=1024)
model = nn.Sequential(nn.Linear(2, 1024), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1024, 1), nn.Sigmoid()) 
print(model)

#data = list(zip(input_data, y))
#random.shuffle(data)

train_data = ExampleData(input_data, y)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

optimizer = Adam(model.parameters(), lr=0.001)

criterion = nn.BCELoss()

loss_list = []

for epoch in range(5):
    for x, y in train_loader:
        # shape x: [4,2]
        optimizer.zero_grad()

        output = model(x).squeeze(dim=1) # shape [4,1] -> shape [4]

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())


plt.figure(figsize=(10, 5))
plt.plot(loss_list, label="Raw Loss", alpha=0.3)
plt.title("Training Loss Over Time")
plt.xlabel("Iteration")
plt.ylabel("BCE Loss")
plt.legend()
plt.savefig("training_loss_dataloader_dropout_0.5.png")







