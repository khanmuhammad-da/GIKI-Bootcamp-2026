import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchinfo import summary

# ==========================================
# DATA
# ==========================================

distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.70], [26.52], [32.47], [37.15], [42.35],
    [46.10], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.80], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=torch.float32)

# ==========================================
# NORMALIZE DATA
# ==========================================

distance_scale = 20.0
time_scale = 100.0

X = distances / distance_scale
y = times / time_scale

# ==========================================
# MLP MODEL
# ==========================================

class MLPLabTask(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)

        return x


# ==========================================
# CREATE MODEL
# ==========================================

model = MLPLabTask()

summary(model, input_size=(39,1))

# ==========================================
# LOSS & OPTIMIZER
# ==========================================

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# TRAINING
# ==========================================

epochs = 3000

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(outputs, y)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1} Loss = {loss.item():.6f}")

# ==========================================
# TESTING
# ==========================================

model.eval()

with torch.no_grad():

    predictions = model(X)

# Convert back to original scale

predictions = predictions * time_scale

print("\nPrediction Results\n")

for d, actual, pred in zip(distances, times, predictions):

    print(
        f"Distance={d.item():5.1f} "
        f"Actual={actual.item():7.2f} "
        f"Predicted={pred.item():7.2f}"
    )

# ==========================================
# PREDICT NEW DATA
# ==========================================

new_distance = torch.tensor([[21.0]])

new_distance_norm = new_distance / distance_scale

with torch.no_grad():

    predicted_time = model(new_distance_norm)

predicted_time = predicted_time * time_scale

print("\nPredicted Time for 21 km = {:.2f}".format(predicted_time.item()))

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(8,5))

plt.scatter(
    distances.numpy(),
    times.numpy(),
    label="Actual Data"
)

plt.plot(
    distances.numpy(),
    predictions.numpy(),
    linewidth=2,
    label="MLP Prediction"
)

plt.xlabel("Distance")
plt.ylabel("Time")
plt.title("Distance vs Time using MLP")

plt.legend()

plt.grid(True)

plt.show()