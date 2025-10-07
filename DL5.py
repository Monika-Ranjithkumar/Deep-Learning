import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define XOR data
# Inputs: 2-bit combinations
X = torch.tensor([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=torch.float32)

# Outputs: XOR labels
Y = torch.tensor([[0],
                  [1],
                  [1],
                  [0]], dtype=torch.float32)

# 2. Define MLP model
class XOR_MLP(nn.Module):
    def __init__(self):
        super(XOR_MLP, self).__init__()
        self.hidden = nn.Linear(2, 4)     # 2 input features → 4 hidden nodes
        self.output = nn.Linear(4, 1)     # 4 hidden nodes → 1 output
        self.activation = nn.Sigmoid()    # Sigmoid activation for both layers

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))  # Final output between 0 and 1
        return x

# 3. Train model
model = XOR_MLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training loop
for epoch in range(5000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# 5. Test model
with torch.no_grad():
    predictions = model(X)
    predicted_labels = (predictions > 0.5).float()
    print("\nPredictions:")
    for i in range(4):
        print(f"Input: {X[i].tolist()} → Predicted: {int(predicted_labels[i].item())}, Target: {int(Y[i].item())}")
