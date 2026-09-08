import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


# ============================================================
# 1. SETUP
# ============================================================

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# Fix random seed for fair comparison
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ============================================================
# 2. LOAD MNIST DATASET
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# 3. RMS NORMALIZATION
# ============================================================

class RMSNorm2D(nn.Module):

    def __init__(self, channels):
        super().__init__()

        # Learnable scale parameter
        self.weight = nn.Parameter(
            torch.ones(1, channels, 1, 1)
        )

        self.eps = 1e-8


    def forward(self, x):

        # x shape:
        # [batch, channels, height, width]

        # Calculate RMS for each sample
        rms = torch.sqrt(
            torch.mean(
                x ** 2,
                dim=(1, 2, 3),
                keepdim=True
            ) + self.eps
        )

        # Normalize
        x = x / rms

        # Learnable scaling
        x = x * self.weight

        return x


# ============================================================
# 4. LAYER NORMALIZATION FOR CNN
# ============================================================

class LayerNorm2D(nn.Module):

    def __init__(self, channels, height, width):

        super().__init__()

        # LayerNorm normalizes C,H,W
        self.norm = nn.LayerNorm(
            [channels, height, width]
        )


    def forward(self, x):

        return self.norm(x)


# ============================================================
# 5. CNN MODEL
# ============================================================

class CNN(nn.Module):

    def __init__(self, norm_type):

        super().__init__()

        # ----------------------------------------------------
        # First convolution
        # Input: 28 x 28
        # After MaxPool: 14 x 14
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            1,
            16,
            kernel_size=3,
            padding=1
        )


        # Select normalization method
        if norm_type == "BatchNorm":

            self.norm1 = nn.BatchNorm2d(16)


        elif norm_type == "LayerNorm":

            self.norm1 = LayerNorm2D(
                16,
                28,
                28
            )


        elif norm_type == "RMSNorm":

            self.norm1 = RMSNorm2D(16)


        elif norm_type == "InstanceNorm":

            self.norm1 = nn.InstanceNorm2d(
                16,
                affine=True
            )


        elif norm_type == "GroupNorm":

            self.norm1 = nn.GroupNorm(
                4,
                16
            )


        # ----------------------------------------------------
        # Second convolution
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1
        )


        if norm_type == "BatchNorm":

            self.norm2 = nn.BatchNorm2d(32)


        elif norm_type == "LayerNorm":

            self.norm2 = LayerNorm2D(
                32,
                14,
                14
            )


        elif norm_type == "RMSNorm":

            self.norm2 = RMSNorm2D(32)


        elif norm_type == "InstanceNorm":

            self.norm2 = nn.InstanceNorm2d(
                32,
                affine=True
            )


        elif norm_type == "GroupNorm":

            self.norm2 = nn.GroupNorm(
                4,
                32
            )


        # ----------------------------------------------------
        # Fully connected layers
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            32 * 7 * 7,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )


        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)


    def forward(self, x):

        # First convolution
        x = self.conv1(x)

        x = self.norm1(x)

        x = self.relu(x)

        x = self.pool(x)

        # Shape:
        # [batch, 16, 14, 14]


        # Second convolution
        x = self.conv2(x)

        x = self.norm2(x)

        x = self.relu(x)

        x = self.pool(x)

        # Shape:
        # [batch, 32, 7, 7]


        # Flatten
        x = x.view(
            x.size(0),
            -1
        )


        # Fully connected
        x = self.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x


# ============================================================
# 6. TRAINING FUNCTION
# ============================================================

def train_model(
    norm_type,
    batch_size,
    epochs=5
):

    print("\n====================================")
    print("Normalization:", norm_type)
    print("Batch Size:", batch_size)
    print("====================================")


    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )


    test_loader = DataLoader(
        test_data,
        batch_size=256,
        shuffle=False
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = CNN(norm_type).to(device)


    # Loss function
    criterion = nn.CrossEntropyLoss()


    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )


    # Store results
    train_losses = []
    test_losses = []
    accuracies = []


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(epochs):

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        model.train()

        total_train_loss = 0


        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)


            # Clear old gradients
            optimizer.zero_grad()


            # Forward pass
            outputs = model(images)


            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )


            # Backpropagation
            loss.backward()


            # Update weights
            optimizer.step()


            total_train_loss += loss.item()


        # Average training loss
        avg_train_loss = (
            total_train_loss /
            len(train_loader)
        )


        # ----------------------------------------------------
        # TESTING / VALIDATION
        # ----------------------------------------------------

        model.eval()

        total_test_loss = 0

        correct = 0
        total = 0


        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)


                outputs = model(images)


                loss = criterion(
                    outputs,
                    labels
                )


                total_test_loss += loss.item()


                # Prediction
                predicted = torch.argmax(
                    outputs,
                    dim=1
                )


                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()


        # Average validation loss
        avg_test_loss = (
            total_test_loss /
            len(test_loader)
        )


        # Accuracy
        accuracy = (
            100 * correct / total
        )


        # Store results
        train_losses.append(
            avg_train_loss
        )

        test_losses.append(
            avg_test_loss
        )

        accuracies.append(
            accuracy
        )


        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {avg_test_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )


    # ========================================================
    # CALCULATE STABILITY
    # ========================================================

    # Standard deviation of training loss
    # Lower = more stable

    stability = np.std(
        train_losses
    )


    # ========================================================
    # CONVERGENCE
    # ========================================================

    # Epoch where minimum training loss occurred

    convergence_epoch = (
        np.argmin(train_losses) + 1
    )


    return {

        "train_loss": train_losses,

        "test_loss": test_losses,

        "accuracy": accuracies,

        "final_train_loss":
            train_losses[-1],

        "final_test_loss":
            test_losses[-1],

        "final_accuracy":
            accuracies[-1],

        "stability":
            stability,

        "convergence_epoch":
            convergence_epoch
    }


# ============================================================
# 7. RUN EXPERIMENT
# ============================================================

normalizations = [

    "BatchNorm",
    "LayerNorm",
    "RMSNorm",
    "InstanceNorm",
    "GroupNorm"

]


batch_sizes = [

    64,
    16,
    4,
    1

]


results = {}


# ------------------------------------------------------------
# Run every combination
# ------------------------------------------------------------

for norm in normalizations:

    for batch in batch_sizes:

        # ----------------------------------------------------
        # BatchNorm with batch size 1
        # ----------------------------------------------------
        #
        # BatchNorm is dependent on batch statistics.
        # If it fails for B=1, we skip it.
        #

        try:

            result = train_model(
                norm,
                batch,
                epochs=5
            )

            results[
                (norm, batch)
            ] = result


        except Exception as e:

            print(
                f"\nSkipped {norm} "
                f"with batch size {batch}"
            )

            print("Reason:", e)


# ============================================================
# 8. CREATE FINAL RESULTS TABLE
# ============================================================

rows = []


for (norm, batch), result in results.items():

    rows.append({

        "Normalization": norm,

        "Batch Size": batch,

        "Final Train Loss":
            result["final_train_loss"],

        "Final Test Loss":
            result["final_test_loss"],

        "Final Accuracy":
            result["final_accuracy"],

        "Stability":
            result["stability"],

        "Convergence Epoch":
            result["convergence_epoch"]

    })


results_df = pd.DataFrame(rows)


print("\n\n====================================")
print("FINAL EXPERIMENT RESULTS")
print("====================================")

print(
    results_df.to_string(
        index=False
    )
)


# Save table
results_df.to_csv(
    "normalization_results.csv",
    index=False
)


# ============================================================
# 9. GRAPH 1 — TRAINING LOSS
# ============================================================

plt.figure(figsize=(10, 6))


# Example: compare all normalization methods
# at batch size = 4

for norm in normalizations:

    if (norm, 4) in results:

        losses = results[
            (norm, 4)
        ]["train_loss"]


        plt.plot(
            range(1, len(losses) + 1),
            losses,
            marker="o",
            label=norm
        )


plt.xlabel("Epoch")

plt.ylabel("Training Loss")

plt.title(
    "Training Loss Comparison - Batch Size 4"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "training_loss_batch4.png"
)

plt.show()


# ============================================================
# 10. GRAPH 2 — VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 6))


for norm in normalizations:

    if (norm, 4) in results:

        losses = results[
            (norm, 4)
        ]["test_loss"]


        plt.plot(
            range(1, len(losses) + 1),
            losses,
            marker="o",
            label=norm
        )


plt.xlabel("Epoch")

plt.ylabel("Validation/Test Loss")

plt.title(
    "Validation Loss Comparison - Batch Size 4"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "validation_loss_batch4.png"
)

plt.show()


# ============================================================
# 11. GRAPH 3 — FINAL ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))


for norm in normalizations:

    accuracies = []
    batches = []


    for batch in batch_sizes:

        if (norm, batch) in results:

            accuracies.append(
                results[
                    (norm, batch)
                ]["final_accuracy"]
            )

            batches.append(batch)


    if len(accuracies) > 0:

        plt.plot(
            batches,
            accuracies,
            marker="o",
            label=norm
        )


plt.xlabel("Batch Size")

plt.ylabel("Final Accuracy (%)")

plt.title(
    "Batch-Size Sensitivity - Final Accuracy"
)

plt.legend()

plt.grid(True)

# Reverse x-axis so 64 → 16 → 4 → 1
plt.gca().invert_xaxis()

plt.tight_layout()

plt.savefig(
    "batch_size_sensitivity.png"
)

plt.show()


# ============================================================
# 12. GRAPH 4 — STABILITY
# ============================================================

plt.figure(figsize=(10, 6))


for norm in normalizations:

    stability_values = []
    batches = []


    for batch in batch_sizes:

        if (norm, batch) in results:

            stability_values.append(
                results[
                    (norm, batch)
                ]["stability"]
            )

            batches.append(batch)


    if len(stability_values) > 0:

        plt.plot(
            batches,
            stability_values,
            marker="o",
            label=norm
        )


plt.xlabel("Batch Size")

plt.ylabel("Training Loss Standard Deviation")

plt.title(
    "Training Stability vs Batch Size"
)

plt.legend()

plt.grid(True)

plt.gca().invert_xaxis()

plt.tight_layout()

plt.savefig(
    "training_stability.png"
)

plt.show()


# ============================================================
# 13. FIND BEST ACCURACY
# ============================================================

best_result = results_df.loc[
    results_df["Final Accuracy"].idxmax()
]


print("\n====================================")
print("BEST FINAL ACCURACY")
print("====================================")

print(
    "Normalization:",
    best_result["Normalization"]
)

print(
    "Batch Size:",
    best_result["Batch Size"]
)

print(
    "Accuracy:",
    f'{best_result["Final Accuracy"]:.2f}%'
)


# ============================================================
# 14. SIMPLE OBSERVATIONS
# ============================================================

print("\n====================================")
print("OBSERVATIONS")
print("====================================")

print("""
1. Training loss should decrease as epochs increase.

2. Lower validation loss indicates better generalization.

3. Higher accuracy indicates better prediction performance.

4. BatchNorm usually performs well with larger batches.

5. BatchNorm can become more sensitive when batch size is very small.

6. LayerNorm and RMSNorm do not depend on batch statistics.

7. InstanceNorm is useful for image/style-related tasks.

8. GroupNorm is a strong choice for CNNs with small batches.

9. Stability is better when training loss has less fluctuation.

10. The best normalization method depends on the task and batch size.
""")


print("\nExperiment completed.")
print("Results saved to normalization_results.csv")
print("Graphs saved as PNG files.")