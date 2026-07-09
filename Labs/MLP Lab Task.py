import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=torch.float32)

#print(distances.shape)
#print(times.shape)

class MLPLabTask(nn.Module):
    def __init__(self):
        super(MLPLabTask, self).__init__()

        self.fc1 = nn.Linear(1,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,4)
        self.fc4 = nn.Linear(4,1)

        #self.dropout1 = nn.Dropout(p=0.2)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        #x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        #x = self.dropout1(x)
        x= torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = MLPLabTask()
from torchinfo import summary
summary(model)


# Train
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
    optimizer.zero_grad()
    out = model(distances)
    #print(out)
    loss = criterion(out,times)
    print(loss.item())
    loss.backward()
    optimizer.step()

    if (epoch+1)%400==0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()

with torch.no_grad():

    predictions = model(distances)


print(predictions)

for d, actual, pred in zip(distances, times, predictions):

    print(
        f"Distance={d.item():4.1f} "
        f"Actual={actual.item():6.2f} "
        f"Predicted={pred.item():6.2f}"
    )