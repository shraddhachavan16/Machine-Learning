import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. SETTINGS
# ============================================================

EPOCHS = 12
LEARNING_RATE = 0.001

BATCH_SIZES = [4, 16, 32, 64]

METHODS = [
    "BatchNorm",
    "LayerNorm",
    "RMSNorm",
    "InstanceNorm",
    "GroupNorm"
]

torch.manual_seed(7)
np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Batch sizes:", BATCH_SIZES)
print("Normalization methods:", METHODS)


# ============================================================
# 2. LOAD DIGITS DATASET
# ============================================================

digits = load_digits()

X = digits.images
y = digits.target

# Convert to float
X = X.astype(np.float32)

# Normalize input pixels
X = X / 16.0

# Add channel dimension
X = X[:, np.newaxis, :, :]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)


# ============================================================
# 3. CNN MODEL
# ============================================================

class NormalizationCNN(nn.Module):

    def __init__(self, norm_type):

        super().__init__()

        self.norm_type = norm_type

        # ----------------------------------------------------
        # First convolution
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # First normalization
        # ----------------------------------------------------

        if norm_type == "BatchNorm":

            self.norm1 = nn.BatchNorm2d(8)

        elif norm_type == "InstanceNorm":

            self.norm1 = nn.InstanceNorm2d(
                8,
                affine=True
            )

        elif norm_type == "GroupNorm":

            self.norm1 = nn.GroupNorm(
                num_groups=4,
                num_channels=8
            )

        # LayerNorm and RMSNorm will be applied after
        # flattening the feature map.

        # ----------------------------------------------------
        # Second convolution
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Second normalization
        # ----------------------------------------------------

        if norm_type == "BatchNorm":

            self.norm2 = nn.BatchNorm2d(16)

        elif norm_type == "InstanceNorm":

            self.norm2 = nn.InstanceNorm2d(
                16,
                affine=True
            )

        elif norm_type == "GroupNorm":

            self.norm2 = nn.GroupNorm(
                num_groups=4,
                num_channels=16
            )

        # ----------------------------------------------------
        # Fully connected layer
        # ----------------------------------------------------

        self.fc = nn.Linear(16 * 2 * 2, 10)

        # LayerNorm / RMSNorm
        #
        # After two pooling operations:
        #
        # 8x8 -> 4x4 -> 2x2
        #
        # Therefore:
        #
        # 16 channels * 2 * 2 = 64
        #

        if norm_type == "LayerNorm":

            self.feature_norm = nn.LayerNorm(64)

        elif norm_type == "RMSNorm":

            self.feature_norm = nn.RMSNorm(64)


    def forward(self, x):

        # ----------------------------------------------------
        # Convolution 1
        # ----------------------------------------------------

        x = self.conv1(x)

        # Apply normalization
        if self.norm_type in [
            "BatchNorm",
            "InstanceNorm",
            "GroupNorm"
        ]:

            x = self.norm1(x)

        x = torch.relu(x)

        # Pool
        x = nn.functional.max_pool2d(x, 2)

        # ----------------------------------------------------
        # Convolution 2
        # ----------------------------------------------------

        x = self.conv2(x)

        # Apply normalization
        if self.norm_type in [
            "BatchNorm",
            "InstanceNorm",
            "GroupNorm"
        ]:

            x = self.norm2(x)

        x = torch.relu(x)

        # Pool
        x = nn.functional.max_pool2d(x, 2)

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        x = torch.flatten(x, start_dim=1)

        # ----------------------------------------------------
        # LayerNorm / RMSNorm
        # ----------------------------------------------------

        if self.norm_type in [
            "LayerNorm",
            "RMSNorm"
        ]:

            x = self.feature_norm(x)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        x = self.fc(x)

        return x


# ============================================================
# 4. TRAINING FUNCTION
# ============================================================

def train_model(norm_type, batch_size):

    print()
    print("=" * 60)
    print("Normalization:", norm_type)
    print("Batch Size:", batch_size)
    print("=" * 60)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Create model
    model = NormalizationCNN(norm_type).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_losses = []
    train_accuracies = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        epoch_loss = running_loss / total

        epoch_accuracy = (
            100 * correct / total
        )

        train_losses.append(epoch_loss)

        train_accuracies.append(
            epoch_accuracy
        )

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )

    # ========================================================
    # TESTING
    # ========================================================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    test_accuracy = (
        100 * correct / total
    )

    print(
        f"Final Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    return {
        "loss": train_losses,
        "accuracy": train_accuracies,
        "test_accuracy": test_accuracy
    }


# ============================================================
# 5. RUN ALL EXPERIMENTS
# ============================================================

results = {}

for batch_size in BATCH_SIZES:

    results[batch_size] = {}

    for method in METHODS:

        results[batch_size][method] = train_model(
            method,
            batch_size
        )


# ============================================================
# 6. PRINT FINAL RESULTS
# ============================================================

print()
print()
print("=" * 80)
print("FINAL TEST ACCURACY COMPARISON")
print("=" * 80)

print(
    f"{'Batch Size':<15}"
    f"{'BatchNorm':<15}"
    f"{'LayerNorm':<15}"
    f"{'RMSNorm':<15}"
    f"{'InstanceNorm':<15}"
    f"{'GroupNorm':<15}"
)

print("-" * 80)

for batch_size in BATCH_SIZES:

    print(
        f"{batch_size:<15}"
        f"{results[batch_size]['BatchNorm']['test_accuracy']:<15.2f}"
        f"{results[batch_size]['LayerNorm']['test_accuracy']:<15.2f}"
        f"{results[batch_size]['RMSNorm']['test_accuracy']:<15.2f}"
        f"{results[batch_size]['InstanceNorm']['test_accuracy']:<15.2f}"
        f"{results[batch_size]['GroupNorm']['test_accuracy']:<15.2f}"
    )


# ============================================================
# 7. GRAPH 1
# TRAINING LOSS FOR DIFFERENT BATCH SIZES
# ============================================================

for batch_size in BATCH_SIZES:

    plt.figure(figsize=(9, 6))

    for method in METHODS:

        plt.plot(
            range(1, EPOCHS + 1),
            results[batch_size][method]["loss"],
            marker="o",
            label=method
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    plt.title(
        f"Training Loss - Batch Size {batch_size}"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"loss_batch_{batch_size}.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 8. GRAPH 2
# TRAINING ACCURACY FOR DIFFERENT BATCH SIZES
# ============================================================

for batch_size in BATCH_SIZES:

    plt.figure(figsize=(9, 6))

    for method in METHODS:

        plt.plot(
            range(1, EPOCHS + 1),
            results[batch_size][method]["accuracy"],
            marker="o",
            label=method
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy (%)")

    plt.title(
        f"Training Accuracy - Batch Size {batch_size}"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"accuracy_batch_{batch_size}.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 9. GRAPH 3
# FINAL TEST ACCURACY FOR ALL BATCH SIZES
# ============================================================

for method in METHODS:

    accuracies = []

    for batch_size in BATCH_SIZES:

        accuracies.append(
            results[batch_size][method]["test_accuracy"]
        )

    plt.plot(
        BATCH_SIZES,
        accuracies,
        marker="o",
        label=method
    )


plt.xlabel("Batch Size")
plt.ylabel("Final Test Accuracy (%)")

plt.title(
    "Effect of Batch Size on Normalization Methods"
)

plt.xticks(BATCH_SIZES)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "batch_size_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. GRAPH 4
# COMPARISON FOR EACH BATCH SIZE
# ============================================================

for batch_size in BATCH_SIZES:

    accuracies = []

    for method in METHODS:

        accuracies.append(
            results[batch_size][method]["test_accuracy"]
        )

    plt.figure(figsize=(9, 6))

    plt.bar(
        METHODS,
        accuracies
    )

    plt.xlabel("Normalization Method")
    plt.ylabel("Final Test Accuracy (%)")

    plt.title(
        f"Normalization Comparison - Batch Size {batch_size}"
    )

    plt.xticks(rotation=20)

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        f"final_accuracy_batch_{batch_size}.png",
        dpi=300
    )

    plt.show()


print()
print("Experiment completed.")
print("Graphs have been saved.")