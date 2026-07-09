import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


X = torch.tensor([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]
])
y = torch.tensor([[0.],[1.],[1.],[0.]])

# Model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,5)
        self.fc3 = nn.Linear(5,1)

        #self.dropout1 = nn.Dropout(p=0.2)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        #x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        #x = self.dropout1(x)

        x = self.fc3(x)
        return x

model = MLP()
from torchinfo import summary
summary(model)

# Train
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2000):
    optimizer.zero_grad()
    out = model(X)
    #print(out)
    loss = criterion(out,y)
    loss.backward()
    optimizer.step()

    if (epoch+1)%400==0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

x = torch.tensor(2.0, requires_grad=True)
y1 = x * x + 3*x + 5
y1.backward()

print(x.grad)

# Test
with torch.no_grad():
    probs = torch.sigmoid(model(X))
    preds = (probs>0.5).float()

print("Probabilities:\n", probs)
print("Classes:\n", preds)