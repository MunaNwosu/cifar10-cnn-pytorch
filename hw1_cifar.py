import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)  # CIFAR: Input size is 32*32*3=3072
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # Output layer with 10 classes (digits 0-9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)  # Dropout with 20% probability
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.softmax(self.fc3(x), dim=1)  # Softmax for 10-way classification
        return x

# Initialize the neural network with He initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

# Create data loaders CIFAR-10
batch_size = 32
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar10_loader = DataLoader(datasets.CIFAR10(
    root='./data', train=True, transform= transform, download=True),
    batch_size=batch_size, shuffle=True)

# Create and initialize the model
model = Net()
model.apply(initialize_weights)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(cifar10_loader):
        optimizer.zero_grad()  # Zero the gradients
        data = data.view(data.size(0), -1)  # Flatten the input data
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Calculate the loss
        loss.backward()  # Back propagation
        optimizer.step()  # Update the weights

    # Print training statistics
    print('Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))

# Evaluation on the test set
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for data, target in cifar10_loader:
        data = data.view(data.size(0), -1)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print('Accuracy on the test set: {:.2f}%'.format(100 * correct / total))

