# 🧠 ML Online 2 Cheatsheet
## PyTorch - Optimization & Neural Networks from Scratch

---

# 📚 TABLE OF CONTENTS
1. [Theory Concepts](#1-theory-concepts)
2. [PyTorch Basics](#2-pytorch-basics)
3. [Optimization Strategies (SGD, Adam)](#3-optimization-strategies-sgd-adam)
4. [Fully Connected Neural Network (FNN)](#4-fully-connected-neural-network-fnn)
5. [Convolutional Neural Network (CNN)](#5-convolutional-neural-network-cnn)
6. [Complete Training Pipeline](#6-complete-training-pipeline)
7. [Common TODO Patterns](#7-common-todo-patterns)
8. [Quick Reference Tables](#8-quick-reference-tables)

---

# 1. THEORY CONCEPTS

## 🎯 What is a Neural Network?
A neural network is a function that maps inputs to outputs through layers of transformations.

```
Input → [Linear Transform] → [Activation] → [Linear Transform] → Output
  X    →      W₁x + b₁      →    ReLU     →      W₂x + b₂      →   ŷ
```

## 🔄 Forward Pass vs Backward Pass

| Pass | What happens | Purpose |
|------|--------------|---------|
| **Forward** | Input flows through network → Output | Make predictions |
| **Backward** | Gradients flow backward | Calculate how to update weights |

## 📉 Gradient Descent - The Core Idea

**Goal**: Minimize the loss function by adjusting weights

```
New Weight = Old Weight - learning_rate × gradient
     W     =     W      -      lr        ×   ∂L/∂W
```

### Example:
```
If loss = 5.0 and gradient = 2.0, lr = 0.1:
New W = W - 0.1 × 2.0 = W - 0.2
→ Weight decreased by 0.2 to reduce loss
```

---

## 🏃 SGD (Stochastic Gradient Descent)

### Formula:
```
θ = θ - lr × ∇L(θ)

Where:
  θ   = parameters (weights)
  lr  = learning rate
  ∇L  = gradient of loss
```

### With Momentum:
```
v = momentum × v + gradient        # accumulate velocity
θ = θ - lr × v                     # update with velocity

Intuition: Like a ball rolling downhill - it gains speed
```

### Example with numbers:
```
Step 1: gradient = 2.0, v = 0, momentum = 0.9, lr = 0.1
        v = 0.9 × 0 + 2.0 = 2.0
        θ = θ - 0.1 × 2.0 = θ - 0.2

Step 2: gradient = 1.5, v = 2.0
        v = 0.9 × 2.0 + 1.5 = 1.8 + 1.5 = 3.3
        θ = θ - 0.1 × 3.3 = θ - 0.33  ← faster update!
```

---

## 🧠 Adam (Adaptive Moment Estimation)

### The Best of Both Worlds:
- **Momentum** (first moment): Remembers direction
- **RMSprop** (second moment): Adapts learning rate per parameter

### Formula:
```
m = β₁ × m + (1 - β₁) × gradient          # momentum (moving average of gradients)
v = β₂ × v + (1 - β₂) × gradient²         # velocity (moving average of squared gradients)

m̂ = m / (1 - β₁ᵗ)                          # bias correction
v̂ = v / (1 - β₂ᵗ)                          # bias correction

θ = θ - lr × m̂ / (√v̂ + ε)                  # update

Default values: β₁=0.9, β₂=0.999, ε=1e-8
```

### Why Adam is Popular:
1. Works well with default hyperparameters
2. Adapts learning rate for each parameter
3. Handles sparse gradients well

---

## 🔥 Activation Functions

| Function | Formula | When to Use | Range |
|----------|---------|-------------|-------|
| **ReLU** | max(0, x) | Hidden layers | [0, ∞) |
| **Sigmoid** | 1/(1+e⁻ˣ) | Binary output | (0, 1) |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Hidden layers | (-1, 1) |
| **Softmax** | eˣⁱ/Σeˣʲ | Multi-class output | (0, 1), sum=1 |

### ReLU Example:
```
Input:  [-2, -1, 0, 1, 2]
ReLU:   [ 0,  0, 0, 1, 2]  ← negative values become 0
```

---

## 🖼️ Convolution Operation

### What it does:
Slides a small filter (kernel) over the image to detect features (edges, patterns)

```
Input Image (5×5)          Kernel (3×3)           Output (3×3)
┌─────────────────┐        ┌─────────┐
│ 1  2  3  4  5   │        │ 1  0  1 │
│ 6  7  8  9  10  │   *    │ 0  1  0 │    =    Feature Map
│ 11 12 13 14 15  │        │ 1  0  1 │
│ 16 17 18 19 20  │        └─────────┘
│ 21 22 23 24 25  │
└─────────────────┘

One calculation: 1×1 + 2×0 + 3×1 + 6×0 + 7×1 + 8×0 + 11×1 + 12×0 + 13×1
              = 1 + 3 + 7 + 11 + 13 = 35
```

### Output Size Formula:
```
Output = (Input - Kernel + 2×Padding) / Stride + 1

Example: Input=28, Kernel=3, Padding=1, Stride=1
Output = (28 - 3 + 2×1) / 1 + 1 = 28
```

---

## 🏊 Pooling Layers

### Max Pooling:
Takes the maximum value in each region → Reduces size, keeps important features

```
Input (4×4)              MaxPool 2×2              Output (2×2)
┌───────────────┐                               ┌───────┐
│ 1  3 │ 2  4  │        Take max of             │ 3   4 │
│ 5  6 │ 7  8  │        each 2×2 region    →    │ 13 16 │
├──────┼───────┤                                └───────┘
│ 9  10│11 12  │
│13 14 │15 16  │
└───────────────┘
```

### Global Average Pooling (GAP):
Takes average of entire feature map → Reduces to 1×1

```
Input (4×4)                              Output (1×1)
┌───────────────┐
│ 1  2  3  4   │
│ 5  6  7  8   │     Average all    →      6.5
│ 9  10 11 12  │
│ 13 14 15 16  │
└───────────────┘
(1+2+...+16)/16 = 136/16 = 8.5
```

---

# 2. PYTORCH BASICS

## 🔧 Essential Imports
```python
import torch                              # Core PyTorch
import torch.nn as nn                     # Neural network modules
import torch.optim as optim               # Optimizers (SGD, Adam)
from torch.utils.data import DataLoader   # Batch loading
from torchvision import datasets, transforms  # Image datasets & transforms
import numpy as np
import random
```

## 🎲 Reproducibility (Important!)
```python
def set_seed(seed=42):
    random.seed(seed)                     # Python random
    np.random.seed(seed)                  # NumPy random
    torch.manual_seed(seed)               # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      # PyTorch GPU

set_seed(42)
```
**Why?** Ensures same random initialization every run → Reproducible results

## 💻 Device Setup
```python
# Check if GPU available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model/tensor to device
model = model.to(device)
tensor = tensor.to(device)
```

---

## 📊 Tensors - The Core Data Structure

### Creating Tensors
```python
# From Python list
x = torch.tensor([1, 2, 3])              # tensor([1, 2, 3])

# Zeros and Ones
x = torch.zeros(3, 4)                    # 3×4 tensor of zeros
x = torch.ones(3, 4)                     # 3×4 tensor of ones

# Random
x = torch.randn(3, 4)                    # 3×4 from normal distribution
x = torch.rand(3, 4)                     # 3×4 from uniform [0, 1)

# From NumPy
x = torch.from_numpy(np_array)
```

### Tensor Properties
```python
x.shape                                  # Size: torch.Size([3, 4])
x.size()                                 # Same as shape
x.dtype                                  # Data type: torch.float32
x.device                                 # Device: cpu or cuda:0
x.requires_grad                          # Track gradients? True/False
```

### Reshaping
```python
x = torch.randn(2, 6)                    # Shape: [2, 6]

x.view(3, 4)                             # Reshape to [3, 4]
x.view(-1, 3)                            # Auto-calculate: [4, 3]
x.view(x.size(0), -1)                    # Flatten: [2, 6] (keep batch dim)

# Example from CNN:
# x has shape [batch_size, 16, 1, 1] after GAP
x = x.view(x.size(0), -1)                # → [batch_size, 16]
# x.size(0) = batch_size, -1 = flatten rest
```

### Matrix Operations
```python
# Matrix multiplication
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B                                # [2, 4]
C = torch.mm(A, B)                       # Same as above
C = torch.matmul(A, B)                   # Same, but handles batches

# Element-wise operations
x + y                                    # Addition
x * y                                    # Element-wise multiply
x ** 2                                   # Square each element
```

---

# 3. OPTIMIZATION STRATEGIES (SGD, Adam)

## 📦 Creating Optimizers

### SGD (Stochastic Gradient Descent)
```python
# Basic SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with Momentum (recommended)
optimizer = optim.SGD(
    model.parameters(),    # Which parameters to optimize
    lr=0.01,               # Learning rate
    momentum=0.9           # Momentum factor (0.9 is common)
)

# SGD with Weight Decay (L2 regularization)
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4      # L2 penalty strength
)
```

### Adam Optimizer
```python
# Basic Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Adam with all parameters
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,              # Learning rate (default: 0.001)
    betas=(0.9, 0.999),    # Coefficients for moving averages
    eps=1e-8,              # Numerical stability
    weight_decay=0         # L2 regularization
)
```

### 📊 SGD vs Adam Comparison

| Feature | SGD | Adam |
|---------|-----|------|
| Learning rate | Fixed for all params | Adaptive per param |
| Default lr | 0.01 - 0.1 | 0.001 |
| Momentum | Optional (add manually) | Built-in |
| Memory | Less | More (stores m, v) |
| Convergence | Slower but generalizes | Faster |

---

## 🔧 Optimizer Methods

### The Training Step
```python
# MUST DO IN THIS ORDER:
optimizer.zero_grad()        # 1. Clear old gradients
loss.backward()              # 2. Compute new gradients
optimizer.step()             # 3. Update weights
```

### Why `zero_grad()`?
```python
# PyTorch ACCUMULATES gradients by default
# Without zero_grad():
# Step 1: grad = 2.0
# Step 2: grad = 2.0 + 1.5 = 3.5  ← WRONG! Accumulated

# With zero_grad():
# Step 1: grad = 0 + 2.0 = 2.0
# Step 2: grad = 0 + 1.5 = 1.5  ← Correct!
```

---

## 🛠️ Custom Optimizer Implementation

### SGD from Scratch
```python
class CustomSGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        # Initialize velocity for each parameter
        self.velocities = [torch.zeros_like(p) for p in self.parameters]
    
    def zero_grad(self):
        """Clear all gradients"""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()     # Set gradients to zero
    
    def step(self):
        """Update parameters"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            if self.momentum > 0:
                # v = momentum * v + gradient
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                # param = param - lr * v
                param.data -= self.lr * self.velocities[i]
            else:
                # Simple SGD: param = param - lr * gradient
                param.data -= self.lr * grad
```

### Adam from Scratch
```python
class CustomAdam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # Time step
        
        # Initialize first moment (m) and second moment (v)
        self.m = [torch.zeros_like(p) for p in self.parameters]
        self.v = [torch.zeros_like(p) for p in self.parameters]
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Update first moment: m = β₁ * m + (1 - β₁) * grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update second moment: v = β₂ * v + (1 - β₂) * grad²
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update: θ = θ - lr * m̂ / (√v̂ + ε)
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

---

# 4. FULLY CONNECTED NEURAL NETWORK (FNN)

## 🏗️ Building an FNN

### Basic Structure
```python
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FNN, self).__init__()
        
        # Layer 1: Input → Hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        # What it does: y = x @ W.T + b
        # W shape: [hidden_size, input_size]
        # b shape: [hidden_size]
        
        # Activation
        self.relu = nn.ReLU()
        # What it does: y = max(0, x)
        
        # Layer 2: Hidden → Output
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: [batch_size, input_size]
        
        out = self.fc1(x)        # [batch_size, hidden_size]
        out = self.relu(out)     # [batch_size, hidden_size]
        out = self.fc2(out)      # [batch_size, num_classes]
        
        return out
```

### Multi-Layer FNN
```python
class DeepFNN(nn.Module):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10):
        super(DeepFNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden1)   # 784 → 256
        self.fc2 = nn.Linear(hidden1, hidden2)      # 256 → 128
        self.fc3 = nn.Linear(hidden2, num_classes)  # 128 → 10
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Randomly zero 50% neurons
    
    def forward(self, x):
        # Flatten if needed: [batch, 1, 28, 28] → [batch, 784]
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)            # Apply dropout
        x = self.relu(self.fc2(x))
        x = self.fc3(x)                # No activation on last layer!
        
        return x
```

### 📐 nn.Linear Explained
```python
# nn.Linear(in_features, out_features)
fc = nn.Linear(4, 3)

# Internally:
# W = random tensor of shape [3, 4]
# b = random tensor of shape [3]

# Forward: output = input @ W.T + b

# Example:
# input shape:  [32, 4]     (batch_size=32, features=4)
# W.T shape:    [4, 3]
# input @ W.T:  [32, 3]
# + b:          [32, 3]     (b is broadcast)
# output shape: [32, 3]
```

---

# 5. CONVOLUTIONAL NEURAL NETWORK (CNN)

## 🏗️ CNN Layers Explained

### nn.Conv2d - Convolution Layer
```python
conv = nn.Conv2d(
    in_channels=1,       # Input channels (1 for grayscale, 3 for RGB)
    out_channels=16,     # Number of filters (feature maps to produce)
    kernel_size=3,       # Filter size (3×3)
    stride=1,            # How many pixels to slide
    padding=1            # Zeros added around border
)

# Example with shapes:
# Input:  [batch, 1, 28, 28]   → 1 channel, 28×28 image
# Output: [batch, 16, 28, 28]  → 16 channels, same size (due to padding=1)

# Output size formula:
# out = ((in + 2*padding - kernel) / stride ) + 1
# out = (28 + 2*1 - 3) / 1 + 1 = 28
```

### nn.MaxPool2d - Max Pooling
```python
pool = nn.MaxPool2d(
    kernel_size=2,       # Pool size (2×2)
    stride=2             # Usually same as kernel_size
)

# Example:
# Input:  [batch, 16, 28, 28]
# Output: [batch, 16, 14, 14]  → Size halved!
```

### nn.AdaptiveAvgPool2d - Global Average Pooling
```python
gap = nn.AdaptiveAvgPool2d((1, 1))
# Takes average of entire feature map
# Input:  [batch, 16, any, any]  → Works for any input size!
# Output: [batch, 16, 1, 1]
```

## batchnormal pooling
```python
self.bn2 = nn.BatchNorm2d(64)
# Normalizes activations across the batch for each channel
```

---

## 🏗️ Complete CNN Example (from demo)

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=1,       # MNIST is grayscale
            out_channels=16,     # 16 feature maps
            kernel_size=3,
            stride=1,
            padding=1
        )
        # Shape change: [batch, 1, 64, 64] → [batch, 16, 64, 64]

        self.relu = nn.ReLU()

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Shape: [batch, 16, 64, 64] → [batch, 16, 1, 1]

        # Fully connected layer
        self.fc = nn.Linear(16, 10)
        # Shape: [batch, 16] → [batch, 10]

    def forward(self, x):
        # x: [batch, 1, 64, 64]
        
        x = self.conv1(x)          # [batch, 16, 64, 64]
        x = self.relu(x)           # [batch, 16, 64, 64]
        
        x = self.gap(x)            # [batch, 16, 1, 1]
        x = torch.flatten(x,1)  # [batch, 16] - Flatten!
        
        x = self.fc(x)             # [batch, 10]
        return x
```

### 📐 Understanding x.view(x.size(0), -1)
```python
# Before: x.shape = [32, 16, 1, 1]
#         batch_size=32, channels=16, height=1, width=1

x = x.view(x.size(0), -1)

x = torch.flatten(x, start_dim=1). # dim=1 since we want to keep batch size intact

# x.size(0) = 32 (batch size)
# -1 = flatten everything else = 16 * 1 * 1 = 16

# After: x.shape = [32, 16]
```

---

## 🏗️ More Complex CNN

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28→28
        self.bn1 = nn.BatchNorm2d(32)        # Normalize activations
        self.pool1 = nn.MaxPool2d(2, 2)      # 28→14
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14→14
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)      # 14→7
        
        # Fully Connected
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)          # [B, 32, 28, 28]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)          # [B, 32, 14, 14]
        
        # Block 2
        x = self.conv2(x)          # [B, 64, 14, 14]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)          # [B, 64, 7, 7]
        
        # Flatten & FC
        x = x.view(x.size(0), -1)  # [B, 64*7*7] = [B, 3136]
        x = self.relu(self.fc1(x)) # [B, 128]
        x = self.dropout(x)
        x = self.fc2(x)            # [B, 10]
        
        return x
```

---

# 6. COMPLETE TRAINING PIPELINE

## 📦 Data Loading

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),              # Resize to 64×64
    transforms.ToTensor(),                     # Convert to tensor [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize (mean, std)
])

# Load dataset
train_dataset = datasets.MNIST(
    root="./data",           # Where to store/find data
    train=True,              # Training set
    download=True,           # Download if not exists
    transform=transform      # Apply transformations
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,             # Test set
    download=True,
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,           # Samples per batch
    shuffle=True             # Shuffle every epoch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False            # No shuffle for testing
)
```

---

## 🎯 Loss Functions

```python
# For multi-class classification (classes: 0, 1, 2, ..., 9)
criterion = nn.CrossEntropyLoss()
# Input: [batch, num_classes] - raw scores (logits)
# Target: [batch] - class indices (0, 1, 2, ...)
# Includes Softmax internally!

# For binary classification
criterion = nn.BCEWithLogitsLoss()
# Input: [batch, 1] - raw scores
# Target: [batch, 1] - 0 or 1
# Includes Sigmoid internally!

# For regression
criterion = nn.MSELoss()  # Mean Squared Error
```

---

## 🔄 Training Loop (Complete)

```python
# Setup
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    # ===== TRAINING PHASE =====
    model.train()                    # Enable training mode (dropout active)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Move data to device
        images = images.to(device)   # [batch, 1, 64, 64]
        labels = labels.to(device)   # [batch]

        # Forward pass
        optimizer.zero_grad()        # Clear gradients
        outputs = model(images)      # [batch, 10]
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()              # Compute gradients
        optimizer.step()             # Update weights

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)  # Get class with highest score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
```

---

## 📊 Evaluation / Inference

```python
model.eval()                         # Disable dropout, batchnorm in eval mode
correct = 0
total = 0

with torch.no_grad():                # Disable gradient computation (faster)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
```

---

## 💾 Saving & Loading Models

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

---

# 7. COMMON TODO PATTERNS

## Pattern 1: Create Optimizer
```python
# TODO: Create SGD optimizer with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# TODO: Create Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Pattern 2: Define Network Layer
```python
# TODO: Add convolution layer
self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)

# TODO: Add fully connected layer
self.fc1 = nn.Linear(in_features=784, out_features=128)

# TODO: Add activation
self.relu = nn.ReLU()
```

## Pattern 3: Forward Pass
```python
# TODO: Apply convolution and activation
x = self.conv1(x)
x = self.relu(x)

# TODO: Flatten the tensor
x = x.view(x.size(0), -1)

# TODO: Apply fully connected layer
x = self.fc(x)
```

## Pattern 4: Training Step
```python
# TODO: Zero the gradients
optimizer.zero_grad()

# TODO: Compute loss
loss = criterion(outputs, labels)

# TODO: Backward pass
loss.backward()

# TODO: Update weights
optimizer.step()
```

## Pattern 5: Calculate Accuracy
```python
# TODO: Get predicted class
_, predicted = outputs.max(1)

# TODO: Count correct predictions
correct += (predicted == labels).sum().item()

# TODO: Calculate accuracy
accuracy = 100.0 * correct / total
```

## Pattern 6: Custom SGD Step
```python
# TODO: Update velocity with momentum
self.velocity = self.momentum * self.velocity + grad

# TODO: Update parameter
param.data = param.data - self.lr * self.velocity
# or
param.data -= self.lr * self.velocity
```

## Pattern 7: Custom Adam Step
```python
# TODO: Update first moment (m)
self.m = self.beta1 * self.m + (1 - self.beta1) * grad

# TODO: Update second moment (v)
self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

# TODO: Bias correction
m_hat = self.m / (1 - self.beta1 ** t)
v_hat = self.v / (1 - self.beta2 ** t)

# TODO: Update parameter
param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

---

# 8. QUICK REFERENCE TABLES

## Layer Syntax

| Layer | Syntax | Input Shape | Output Shape |
|-------|--------|-------------|--------------|
| Linear | `nn.Linear(in, out)` | [B, in] | [B, out] |
| Conv2d | `nn.Conv2d(c_in, c_out, k)` | [B, c_in, H, W] | [B, c_out, H', W'] |
| MaxPool2d | `nn.MaxPool2d(k)` | [B, C, H, W] | [B, C, H/k, W/k] |
| AdaptiveAvgPool2d | `nn.AdaptiveAvgPool2d((h,w))` | [B, C, H, W] | [B, C, h, w] |
| BatchNorm2d | `nn.BatchNorm2d(channels)` | [B, C, H, W] | [B, C, H, W] |
| Dropout | `nn.Dropout(p)` | Any | Same |

## Activation Functions

| Function | Syntax | Use Case |
|----------|--------|----------|
| ReLU | `nn.ReLU()` | Hidden layers |
| Sigmoid | `nn.Sigmoid()` | Binary output |
| Tanh | `nn.Tanh()` | Hidden layers |
| Softmax | `nn.Softmax(dim=1)` | Multi-class output |

## Optimizer Comparison

| Optimizer | Syntax | Default LR | When to Use |
|-----------|--------|------------|-------------|
| SGD | `optim.SGD(params, lr)` | 0.01-0.1 | Simple, good generalization |
| SGD+M | `optim.SGD(params, lr, momentum=0.9)` | 0.01-0.1 | Faster than vanilla SGD |
| Adam | `optim.Adam(params, lr)` | 0.001 | Works well out-of-box |

## Common Transforms

| Transform | Syntax | Purpose |
|-----------|--------|---------|
| Resize | `transforms.Resize((H, W))` | Change image size |
| ToTensor | `transforms.ToTensor()` | PIL → Tensor, scale to [0,1] |
| Normalize | `transforms.Normalize(mean, std)` | Standardize values |
| Compose | `transforms.Compose([...])` | Chain transforms |

---

# 🚀 COMPLETE TEMPLATE

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ===== 1. SETUP =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. DATA =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ===== 3. MODEL =====
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def __init__(self):
        super(Net,self).__init__()
        # 28 x 28 input images, 3 channel (grayscale)
        self.features =  nn.Sequential(
            nn.Conv2d(3, 32, 3, 1), # Bx32x26x26
            nn.BatchNorm2d(32),  
            nn.Relu(),
            nn.maxPool2d(2,2) ,  #Bx32x13x13
            # channel count previously ja chlo ( 2nd dim )
            nn.Conv2d(32, 64, 3,2 ) # Bx64x6x6
            nn.BatchNorm2d(64),
            nn.Relu(),
            nn.maxPool2d(2,2)  # Bx64x3x3

        )
        self.classifier = nn.Sequential(
            nn.Linear(64*3*3,128),
            nn.Relu(),
            nn.Linear(128,10)

        )
        def forward(self,x):
            x = self.features(x)
            x = x.view(x.size(0), -1) # flatten
            x = self.classifier (x)
            return x

model = Net().to(device)

# ===== 4. LOSS & OPTIMIZER =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== 5. TRAINING =====
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} complete")

print("Training finished!")
```

---

# ⚡ QUICK RECALL TIPS

1. **Always call `optimizer.zero_grad()` BEFORE `loss.backward()`**

2. **Order: zero_grad → forward → loss → backward → step**

3. **`model.train()` for training, `model.eval()` for testing**

4. **Use `with torch.no_grad():` during evaluation**

5. **Flatten before FC layer: `x.view(x.size(0), -1)`**

6. **CrossEntropyLoss includes Softmax - don't add it in model!**

7. **Adam default lr=0.001, SGD default lr=0.01**

8. **Adam formula: m, v, then bias correction, then update**

9. **SGD momentum: v = momentum * v + grad**

10. **Output size after Conv: (in + 2*pad - kernel) / stride + 1**

---

Good luck with your exam! 🍀
