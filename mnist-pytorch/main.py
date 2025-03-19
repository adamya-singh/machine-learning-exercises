#videos used:
# cross entropy: https://youtu.be/Pwgpl9mKars?si=UOsmvzw9ltQ0Nwko, https://youtu.be/SxGYPqCgJWM?si=8T-auvtgKCpjnZZJ
# optimizers: https://youtu.be/NE88eqLngkg?si=QWkMQjQJXYEH7WBM
# 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=my_transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=my_transform)

class babysFirstNet(nn.Module):
    def __init__(self):
        super(babysFirstNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

trainloader = DataLoader(trainset, batch_size=10, shuffle=True)
testloader = DataLoader(testset, batch_size=10, shuffle=False)

model = babysFirstNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #print(f"running loss: {running_loss}")
        print(f"step loss: {loss.data}")
    print(f"Epoch {epoch +1}, Loss: {running_loss/len(trainloader)}")


#testing
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")