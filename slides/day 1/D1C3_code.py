#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAT41130 — AI for Weather & Climate
Day 1 Companion Script (Lectures: Linear Regression → Neural Networks, forward-pass focus)

Last updated: 2025-09-18

What this script is:
    A heavily commented, line-by-line teaching companion for the first two lectures.
    It takes you from:
        * dot products and linear transforms,
        * to linear regression as a single-neuron network,
        * to activations (sigmoid, tanh, ReLU),
        * to multi-feature inputs and matrix shapes,
        * and finally to a small taste of PyTorch and autograd.

How to use:
    - Run this script top-to-bottom. Skim comments first, then re-run slowly, line-by-line.
    - Search for the tag 'EXERCISE' — those blocks are designed to pause & discuss.
    - You can tweak knobs (learning rate, epochs, initial weights) and see the effect.
    - Plots are optional but nice; they use matplotlib (no custom styles set, per course rules).

Why so many comments?
    The aim is clarity. Every step is annotated so you can connect the lecture slides to code.

Note:
    We intentionally do not use PyTorch until Section 10 to ensure you understand the core math.
    Early sections use only Python and NumPy.

"""

# =============================================================================
# Section 0 — Imports, printing helpers, reproducibility
# =============================================================================

from __future__ import annotations

import sys
import math
import random
from dataclasses import dataclass
from typing import Tuple, Callable, Iterable, List, Optional

import numpy as np

# Matplotlib is used for a few simple 1-figure plots (one chart per figure; no custom colors).
import matplotlib.pyplot as plt

# We will only import torch later (Section 10) so that the early parts are framework-free.

# For reproducibility in random demos:
np.random.seed(42)
random.seed(42)


def header(title: str):
    """Pretty-print a section header so it's obvious in the console."""
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}\n")


def subheader(title: str):
    """Pretty-print a sub-section header."""
    line = "-" * len(title)
    print(f"\n{title}\n{line}")


def show_vector(name: str, v: np.ndarray):
    """Display a 1-D vector with its shape (convenience for teaching)."""
    print(f"{name} (shape {v.shape}): {np.array2string(v, precision=4, floatmode='fixed')}")


def show_matrix(name: str, M: np.ndarray):
    """Display a 2-D matrix with its shape (convenience for teaching)."""
    with np.printoptions(precision=4, suppress=True, floatmode='fixed'):
        print(f"{name} (shape {M.shape}):\n{M}")


# =============================================================================
# Section 1 — The tiniest 'weather' dataset & dot product warm‑up
# =============================================================================

header("Section 1 — Tiny 'weather' dataset & dot product")

# We'll reuse the 5-point temperature example from the lecture slides:
# X = temperature yesterday, y = temperature today
X = np.array([16.09, 15.56, 15.85, 15.69, 15.01], dtype=float)
y = np.array([17.62, 14.88, 16.32, 16.28, 14.96], dtype=float)

show_vector("X (yesterday °C)", X)
show_vector("y (today °C)", y)

# Dot product refresher (forward step inside a neuron):
x = np.array([2.0, -3.0, 0.5])  # toy 3-d input
w = np.array([0.1, 0.2, 0.3])   # toy weights
b = 0.05                        # bias

dot = x @ w  # same as np.dot(x, w)
z = dot + b  # pre-activation (a.k.a. linear transform)
print(f"\nToy dot product: x @ w = {dot:.4f}, plus bias => z = {z:.4f}")

# EXERCISE (verbal): Change x, w, b and predict how z changes before running the code.


# =============================================================================
# Section 2 — Linear regression as a single-neuron network
# =============================================================================

header("Section 2 — Linear regression as a single neuron")

# A linear regression with 1 feature can be seen as:
# y_hat = w * X + b  where w,b are parameters to learn.

def predict_lr_1d(X: np.ndarray, w: float, b: float) -> np.ndarray:
    """Forward pass for 1D linear regression."""
    return w * X + b


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — often interpretable in the same units as y."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# Let's try the two candidates from the slides to build intuition.
candidates = [
    dict(w=0.0, b=0.5),
    dict(w=1.0, b=1.0),
]

for c in candidates:
    yhat = predict_lr_1d(X, c["w"], c["b"])
    loss = rmse(y, yhat)
    print(f"Candidate (w={c['w']:.2f}, b={c['b']:.2f}) → RMSE = {loss:.4f}")

# The second pair should be dramatically better on these 5 points (as shown in class).


# =============================================================================
# Section 3 — Visual intuition: plotting the fit for a few (w, b)
# =============================================================================

header("Section 3 — Visualising a few candidate lines")

def plot_candidates(X: np.ndarray, y: np.ndarray, wb_list: List[Tuple[float, float]]):
    """
    Show X vs y and a few candidate regression lines.
    Rule for plots in this course:
        * matplotlib only
        * one chart per figure
        * do not set any specific colors or styles
    """
    plt.figure()
    plt.scatter(X, y, label="Data")
    xgrid = np.linspace(X.min()-0.5, X.max()+0.5, 100)
    for w, b in wb_list:
        plt.plot(xgrid, w * xgrid + b, label=f"w={w:.2f}, b={b:.2f}")
    plt.xlabel("Temperature yesterday (°C)")
    plt.ylabel("Temperature today (°C)")
    plt.title("Linear regression — candidate fits")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_candidates(X, y, wb_list=[(0.0, 0.5), (1.0, 1.0), (1.2, 0.0)])


# =============================================================================
# Section 4 — Loss landscapes by brute force (grid search)
# =============================================================================

header("Section 4 — Loss landscape via grid search")

def grid_search_rmse(X: np.ndarray, y: np.ndarray,
                     w_values: np.ndarray, b_values: np.ndarray) -> np.ndarray:
    """Return a 2D array L[i,j] = RMSE at w[i], b[j]."""
    L = np.zeros((len(w_values), len(b_values)), dtype=float)
    for i, w in enumerate(w_values):
        for j, b in enumerate(b_values):
            L[i, j] = rmse(y, predict_lr_1d(X, w, b))
    return L


w_grid = np.linspace(0.0, 2.0, 41)  # 41 points in [0, 2]
b_grid = np.linspace(-1.0, 1.0, 41)
L = grid_search_rmse(X, y, w_grid, b_grid)

# Visualize as a contour plot (one figure, no custom colors).
plt.figure()
W, B = np.meshgrid(w_grid, b_grid, indexing="ij")
CS = plt.contour(W, B, L, levels=15)
plt.clabel(CS, inline=True, fontsize=8)
plt.xlabel("w")
plt.ylabel("b")
plt.title("RMSE loss landscape — 1D linear regression")
plt.tight_layout()
plt.show()


# =============================================================================
# Section 5 — Gradient descent for 1D linear regression (by hand)
# =============================================================================

header("Section 5 — Gradient descent for 1D LR (by hand)")

def compute_gradients_1d(X: np.ndarray, y: np.ndarray, w: float, b: float) -> Tuple[float, float]:
    """
    Compute gradients of MSE loss (not RMSE) w.r.t w and b.
    We can use MSE because it's simpler for derivatives; updating under MSE or RMSE is similar near optimum.
    d/dw MSE = (2/n) * sum( (w*X + b - y) * X )
    d/db MSE = (2/n) * sum( (w*X + b - y) )
    """
    n = len(X)
    yhat = w * X + b
    residual = yhat - y
    dw = (2.0 / n) * np.sum(residual * X)
    db = (2.0 / n) * np.sum(residual)
    return float(dw), float(db)


def gradient_descent_1d(X: np.ndarray, y: np.ndarray, w0: float, b0: float,
                        lr: float = 0.01, epochs: int = 100) -> Tuple[float, float, List[float]]:
    """Run gradient descent and return final (w, b, loss_history)."""
    w, b = float(w0), float(b0)
    losses = []
    for epoch in range(epochs):
        dw, db = compute_gradients_1d(X, y, w, b)
        w -= lr * dw
        b -= lr * db
        loss = rmse(y, w * X + b)
        losses.append(loss)
        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:4d} | RMSE: {loss:.6f} | w: {w:.6f} | b: {b:.6f}")
    return w, b, losses


# Initialize at zeros:
w_init, b_init = 0.0, 0.0
w_hat, b_hat, loss_hist = gradient_descent_1d(X, y, w_init, b_init, lr=0.01, epochs=120)

# Plot the training curve (single figure).
plt.figure()
plt.plot(np.arange(len(loss_hist)), loss_hist)
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Gradient descent learning curve (1D LR)")
plt.tight_layout()
plt.show()


# =============================================================================
# Section 6 — From 1 feature to many: vector & matrix forms
# =============================================================================

header("Section 6 — Multi-feature LR & shapes")

# Let's construct a toy feature matrix with 2 inputs (e.g., temperature and humidity yesterday).
# We'll create tiny synthetic data to keep focus on the shapes.
n_samples = 5
X2 = np.column_stack([
    X,                                  # feature 1 = temperature yesterday
    np.array([65, 70, 68, 72, 60], float)  # feature 2 = humidity yesterday (%)
])  # shape: (5, 2)

# We'll pretend "today's temperature" depends on both features linearly.
# For demo, set up pseudo-true parameters:
w_true = np.array([1.02, -0.03])   # weights for (temp, humidity)
b_true = 0.06                      # small bias

y2 = X2 @ w_true + b_true  # forward pass in vector form
show_matrix("X2", X2)
show_vector("w_true", w_true)
print(f"b_true: {b_true:.4f}")
show_vector("y2 = X2 @ w_true + b_true", y2)

# EXERCISE: Check shapes. X2 is (5,2); w_true is (2,); bias is scalar. Why does broadcasting work here?

def predict_lr_multi(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Vectorized forward pass for multi-feature linear regression."""
    return X @ w + b


# =============================================================================
# Section 7 — Binary classification: logistic regression forward
# =============================================================================

header("Section 7 — Binary classification & sigmoid")

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary cross-entropy loss.
    We clip probabilities for numerical stability to avoid log(0).
    """
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


# Build a tiny synthetic binary dataset with two features (x1, x2), and a linearly separable boundary.
rng = np.random.default_rng(0)
N = 60
x1 = rng.normal(loc=0.0, scale=1.0, size=N)
x2 = rng.normal(loc=0.0, scale=1.0, size=N)
Xb = np.column_stack([x1, x2])
# True underlying separator: 0.7 * x1 - 0.5 * x2 + 0.2 > 0 ⇒ class 1
ybin = (0.7 * x1 - 0.5 * x2 + 0.2 > 0).astype(float)

# Forward pass with random weights:
w_log = rng.normal(size=2)
b_log = 0.0
z = Xb @ w_log + b_log
p = sigmoid(z)
bce = binary_cross_entropy(ybin, p)
print(f"Random-init logistic regression → BCE = {bce:.4f}")

# NOTE: We aren't training yet — just connecting the forward step to probabilities in (0,1).


# =============================================================================
# Section 8 — Activation functions: sigmoid, tanh, ReLU (by hand)
# =============================================================================

header("Section 8 — Activation functions in action")

def relu(z: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0.0, z)


def tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent."""
    return np.tanh(z)


# Demo: apply activations to a simple pre-activation vector
z_demo = np.linspace(-3, 3, 13)
show_vector("z_demo", z_demo)
show_vector("sigmoid(z_demo)", sigmoid(z_demo))
show_vector("tanh(z_demo)", tanh(z_demo))
show_vector("relu(z_demo)", relu(z_demo))

# Plot the three activation curves on separate figures (course rule: one chart per figure).
plt.figure()
plt.plot(z_demo, sigmoid(z_demo))
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.title("Sigmoid activation")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(z_demo, tanh(z_demo))
plt.xlabel("z")
plt.ylabel("tanh(z)")
plt.title("Tanh activation")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(z_demo, relu(z_demo))
plt.xlabel("z")
plt.ylabel("ReLU(z)")
plt.title("ReLU activation")
plt.tight_layout()
plt.show()


# =============================================================================
# Section 9 — A tiny 2-layer network: manual forward pass
# =============================================================================

header("Section 9 — Manual forward for a tiny 2-layer NN")

# We'll build a 2-2-1 network (2 inputs → 2 hidden (ReLU) → 1 output (linear)).
# Shapes:
#   W1: (2, 2), b1: (2,)
#   W2: (2, 1), b2: (1,)

def forward_2_2_1(x: np.ndarray,
                  W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray) -> float:
    """
    Manual forward pass for one sample x (shape: (2,)).
    """
    h_pre = x @ W1 + b1           # shape (2,)
    h = relu(h_pre)               # activation
    yhat = h @ W2 + b2            # scalar
    return float(yhat)


# Make up a simple example:
x_sample = np.array([0.3, -1.2])
W1 = np.array([[ 0.5, -0.4],
               [ 1.1,  0.2]])
b1 = np.array([0.0, -0.1])
W2 = np.array([[ 0.7],
               [-1.0]])
b2 = np.array([0.05])

yhat_ex = forward_2_2_1(x_sample, W1, b1, W2, b2)
print(f"Manual forward output for x={x_sample}: y_hat = {yhat_ex:.4f}")

# EXERCISE: Zero out b1 and b2. Re-run. What's the effect of removing biases?


# =============================================================================
# Section 10 — Enter PyTorch: tensors, autograd, and fitting LR
# =============================================================================

header("Section 10 — PyTorch: tensors & autograd (linear regression)")

# We delay importing torch until now to keep early sections math-first.
import torch
torch.manual_seed(42)

# Convert our original small dataset to torch tensors.
X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # shape (5,1)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (5,1)

# Define a minimal linear model: y = w*X + b
model_lr = torch.nn.Linear(in_features=1, out_features=1, bias=True)

# Mean Squared Error loss and a basic optimizer (SGD).
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model_lr.parameters(), lr=0.05)

# Train loop — VERY small, for demonstration.
epochs = 200
losses_torch = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model_lr(X_t)
    loss = criterion(y_pred, y_t)
    loss.backward()       # autograd computes d(loss)/d(params)
    optimizer.step()      # update parameters
    losses_torch.append(float(torch.sqrt(loss).item()))  # store RMSE
    if epoch % 20 == 0:
        w_val = model_lr.weight.item()
        b_val = model_lr.bias.item()
        print(f"[Torch LR] Epoch {epoch:3d} | RMSE ~ {math.sqrt(loss.item()):.6f} | w: {w_val:.6f} | b: {b_val:.6f}")

# Plot the PyTorch learning curve (single figure).
plt.figure()
plt.plot(np.arange(len(losses_torch)), losses_torch)
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("PyTorch linear regression — learning curve")
plt.tight_layout()
plt.show()

# Inspect learned parameters.
w_learned = model_lr.weight.detach().cpu().numpy().squeeze()
b_learned = model_lr.bias.detach().cpu().numpy().squeeze()
print(f"Learned parameters (PyTorch): w ≈ {w_learned:.4f}, b ≈ {b_learned:.4f}")


# =============================================================================
# Section 11 — PyTorch: a tiny MLP for binary classification
# =============================================================================

header("Section 11 — PyTorch MLP (2→3→1) with ReLU on synthetic binary data")

# We'll reuse Xb (N x 2) and ybin (N) from Section 7 but convert to tensors.
Xb_t = torch.tensor(Xb, dtype=torch.float32)
ybin_t = torch.tensor(ybin, dtype=torch.float32).unsqueeze(1)

class TinyMLP(torch.nn.Module):
    """
    A tiny multilayer perceptron:
        input 2 → hidden 3 (ReLU) → output 1 (logit)
    We'll use BCEWithLogitsLoss which expects raw scores ("logits"),
    and combines sigmoid + BCE in a stable way.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 3)
        self.fc2 = torch.nn.Linear(3, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        logit = self.fc2(h)
        return logit

mlp = TinyMLP()
criterion_bce = torch.nn.BCEWithLogitsLoss()
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.05)

epochs = 300
loss_hist_mlp = []
for epoch in range(epochs):
    optimizer_mlp.zero_grad()
    logits = mlp(Xb_t)
    loss = criterion_bce(logits, ybin_t)
    loss.backward()
    optimizer_mlp.step()
    loss_hist_mlp.append(float(loss.item()))
    if epoch % 50 == 0:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            acc = (preds.eq(ybin_t).float().mean().item())
        print(f"[MLP] Epoch {epoch:3d} | loss: {loss.item():.4f} | acc: {acc:.3f}")

# Plot training loss (single figure).
plt.figure()
plt.plot(np.arange(len(loss_hist_mlp)), loss_hist_mlp)
plt.xlabel("Epoch")
plt.ylabel("BCE loss")
plt.title("Tiny MLP — training loss")
plt.tight_layout()
plt.show()


# =============================================================================
# Section 12 — Sanity checks & common pitfalls (from the slides)
# =============================================================================

header("Section 12 — Sanity checks & pitfalls")

checks = [
    "Check shapes at every layer (especially batch dimension).",
    "Do not forget the bias terms.",
    "Use appropriate activation for the task (e.g., none/linear for regression, sigmoid/logits for binary).",
    "Normalize or standardize inputs when features are on very different scales.",
    "Monitor a validation split if data volume allows; early stopping helps.",
    "Learning rate matters. Too big → diverge; too small → crawl.",
]
for i, c in enumerate(checks, 1):
    print(f"{i}. {c}")

# EXERCISE: Print the parameter counts for TinyMLP and discuss how they scale.
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"TinyMLP trainable parameters: {count_parameters(mlp)}")


# =============================================================================
# Section 13 — Mini-lab ideas (optional explorations)
# =============================================================================

header("Section 13 — Mini-lab ideas")

ideas = [
    "Swap ReLU for tanh in the hidden layer. What changes?",
    "Standardize x1,x2 before feeding the MLP; rerun and compare convergence.",
    "Create a new synthetic decision boundary and see if the MLP can learn it.",
    "Back in Section 5, try different learning rates (1e-3 to 1e-1) and plot learning curves.",
    "Add L2 weight decay in the PyTorch optimizers (weight_decay=1e-4) and note effects.",
]
for i, idea in enumerate(ideas, 1):
    print(f"({i}) {idea}")

print("\nAll done 🎉  You have stepped from dot products → linear models → activations → tiny NNs,")
print("and you've seen both from-scratch math and PyTorch's autograd in action.")

