
# CS 228 – Biometric Security with AI
# Assignment 2: Multiclass Clean-Label Data Poisoning
# Clean-label poisoning attack on CIFAR-10 (4-class subset)
# Based on the "Poison Frogs" methodology (Shafahi et al., 2018)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from torch.utils.data import DataLoader, Subset, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# 0. Device Setup
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device('mps')
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Preparation
# Select 4 classes from CIFAR-10: Airplane(0), Automobile(1), Bird(2), Cat(3)
# Use 500 images per class for training.
# Remove 10 target images (class 0 = Airplane) from training set.
# Pick 10 base images (class 1 = Automobile).
# All poison labels remain the BASE class label.
# ─────────────────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
])

print("\n[1/5] Loading CIFAR-10 data...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)

CLASS_NAMES  = ['Airplane', 'Automobile', 'Bird', 'Cat']
CLASSES      = [0, 1, 2, 3]
SAMPLES_PER_CLASS = 500
TARGET_CLASS = 0   # Airplane
BASE_CLASS   = 1   # Automobile
NUM_POISONS  = 10

def filter_dataset(dataset, classes, samples_per_class=None):
    """Return a Subset containing at most samples_per_class images per class."""
    indices     = []
    class_counts = {c: 0 for c in classes}
    for i, (_, label) in enumerate(dataset):
        if label in classes:
            if samples_per_class is None or class_counts[label] < samples_per_class:
                indices.append(i)
                class_counts[label] += 1
            if samples_per_class is not None and all(
                    v == samples_per_class for v in class_counts.values()):
                break
    return Subset(dataset, indices)

train_subset = filter_dataset(trainset, CLASSES, SAMPLES_PER_CLASS)
test_subset  = filter_dataset(testset,  CLASSES, None)

print(f"  Training subset size : {len(train_subset)}")
print(f"  Test subset size     : {len(test_subset)}")

# Extract into tensors
train_imgs, train_lbls = [], []
for img, lbl in train_subset:
    train_imgs.append(img)
    train_lbls.append(lbl)

train_imgs = torch.stack(train_imgs)
train_lbls = torch.tensor(train_lbls)

# Separate target and base images
target_indices = (train_lbls == TARGET_CLASS).nonzero(as_tuple=True)[0][:NUM_POISONS]
base_indices   = (train_lbls == BASE_CLASS  ).nonzero(as_tuple=True)[0][:NUM_POISONS]

targets = train_imgs[target_indices].clone()   # 10 airplane images (held out)
bases   = train_imgs[base_indices  ].clone()   # 10 automobile images (poison seeds)

# Remove target images from training set
mask = torch.ones(len(train_imgs), dtype=torch.bool)
mask[target_indices] = False
clean_train_imgs = train_imgs[mask]
clean_train_lbls = train_lbls[mask]

clean_train_dataset = TensorDataset(clean_train_imgs, clean_train_lbls)
clean_train_loader  = DataLoader(clean_train_dataset, batch_size=64, shuffle=True)
test_loader         = DataLoader(test_subset, batch_size=64, shuffle=False)

print(f"  Clean training set (targets removed): {len(clean_train_dataset)}")
print(f"  Target images (held out)            : {len(targets)}")
print(f"  Base images (poison seeds)          : {len(bases)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Architecture
# Small CNN: 3 conv layers + 2 FC layers, outputs logits for 4 classes.
# get_features() extracts the penultimate (256-dim) representation.
# ─────────────────────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def get_features(self, x):
        """Penultimate-layer (256-dim) representation."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Initial Training – train on CLEAN set (targets excluded)
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, class_names=CLASS_NAMES):
    model.eval()
    correct, total = 0, 0
    per_class_correct = [0] * len(class_names)
    per_class_total   = [0] * len(class_names)

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
            for i in range(lbls.size(0)):
                lbl = lbls[i].item()
                per_class_correct[lbl] += (preds[i] == lbls[i]).item()
                per_class_total[lbl]   += 1

    acc = 100.0 * correct / total
    print(f"  Overall accuracy: {acc:.2f}%")
    for i, name in enumerate(class_names):
        if per_class_total[i]:
            print(f"    Class {i} ({name}): "
                  f"{100*per_class_correct[i]/per_class_total[i]:.2f}%")
    return acc, per_class_correct, per_class_total


print("\n[2/5] Training initial CLEAN model (20 epochs)...")
model     = SimpleCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
for ep in range(EPOCHS):
    loss = train_epoch(model, clean_train_loader, optimizer, criterion)
    if (ep + 1) % 5 == 0 or ep == 0:
        print(f"  Epoch {ep+1:2d}/{EPOCHS}  loss={loss:.4f}")

print("\n  Evaluating clean model on test set:")
clean_acc, clean_per_class_correct, clean_per_class_total = evaluate(model, test_loader)

# Check predictions on held-out target images BEFORE poisoning
model.eval()
targets_dev = targets.to(device)
with torch.no_grad():
    target_preds_clean = model(targets_dev).argmax(dim=1).cpu().numpy()
print(f"\n  Target images predicted as (clean model): {target_preds_clean}")
print(f"  (Expected mostly {TARGET_CLASS} = {CLASS_NAMES[TARGET_CLASS]})")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Poison Generation – Poison Frogs iterative algorithm
#
# For each (target_i, base_i) pair:
#   - Initialize poison x = base_i
#   - Iterate:
#       Forward  step : gradient descent on x to minimise ||f(x) - f(t)||^2
#       Backward step : project x back toward base_i via Frobenius constraint
#   - Clamp pixel values to [0, 1]
#
# Visualise at every 50 iterations (5 snapshots per poison).
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Generating poison images...")

BETA       = 0.05    # visual similarity weight (lower → closer to target feature)
LR_POISON  = 0.05    # Adam LR for poison optimisation
ITERATIONS = 300     # number of optimisation steps per poison

model.eval()  # freeze model during poison generation

poisons = []

# Figure: evolution grid (NUM_POISONS rows × 6 cols)
# cols: [base | iter50 | iter100 | iter150 | iter300 | target]
SNAP_ITERS = [50, 100, 150, ITERATIONS]
N_COLS = 2 + len(SNAP_ITERS)          # base + snapshots + target

fig_evo, axes_evo = plt.subplots(NUM_POISONS, N_COLS,
                                  figsize=(3 * N_COLS, 3 * NUM_POISONS))
if NUM_POISONS == 1:
    axes_evo = axes_evo[np.newaxis, :]

for i in range(NUM_POISONS):
    t_img = targets[i].unsqueeze(0).to(device)
    b_img = bases[i].unsqueeze(0).to(device)

    # Initialise poison as a clone of the base image (requires grad)
    p_img = b_img.clone().detach().requires_grad_(True)

    opt_p = optim.Adam([p_img], lr=LR_POISON)
    t_feat = model.get_features(t_img).detach()

    # Column 0: base image
    axes_evo[i, 0].imshow(
        np.transpose(b_img.squeeze().cpu().numpy(), (1, 2, 0)).clip(0, 1))
    axes_evo[i, 0].set_title("Base", fontsize=8)
    axes_evo[i, 0].axis('off')

    snap_col = 1
    for it in range(1, ITERATIONS + 1):
        opt_p.zero_grad()

        p_feat     = model.get_features(p_img)
        loss_feat  = torch.norm(p_feat - t_feat, p=2) ** 2   # forward step
        loss_base  = torch.norm(p_img  - b_img,  p=2) ** 2   # backward step
        loss       = loss_feat + BETA * loss_base

        loss.backward()
        opt_p.step()

        with torch.no_grad():
            p_img.clamp_(0.0, 1.0)

        if it in SNAP_ITERS and snap_col < N_COLS - 1:
            axes_evo[i, snap_col].imshow(
                np.transpose(p_img.detach().squeeze().cpu().numpy(),
                             (1, 2, 0)).clip(0, 1))
            axes_evo[i, snap_col].set_title(f"Iter {it}", fontsize=8)
            axes_evo[i, snap_col].axis('off')
            snap_col += 1

    # Column last: target image
    axes_evo[i, -1].imshow(
        np.transpose(t_img.squeeze().cpu().numpy(), (1, 2, 0)).clip(0, 1))
    axes_evo[i, -1].set_title("Target", fontsize=8)
    axes_evo[i, -1].axis('off')

    final_feat_dist = torch.norm(
        model.get_features(p_img).detach() - t_feat).item()
    print(f"  Poison {i+1:2d}/{NUM_POISONS}  "
          f"feat_dist={final_feat_dist:.4f}  "
          f"base_dist={torch.norm(p_img.detach()-b_img).item():.4f}")

    poisons.append(p_img.detach().cpu().squeeze(0))

fig_evo.suptitle("Poison Evolution (Base → Iterations → Target)", fontsize=12)
fig_evo.tight_layout()
fig_evo.savefig("poisons_evolution.png", dpi=100, bbox_inches='tight')
plt.close(fig_evo)
print("  Saved: poisons_evolution.png")

poisons = torch.stack(poisons)   # shape (10, 3, 32, 32)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Retraining & Evaluation
# Inject poisons (labelled as BASE class) into the training set.
# Retrain a FRESH model from scratch on the poisoned dataset.
# Evaluate overall accuracy and attack success rate.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Retraining from scratch on poisoned dataset...")

poison_labels = torch.full((NUM_POISONS,), BASE_CLASS, dtype=torch.long)
poisoned_imgs = torch.cat([clean_train_imgs, poisons])
poisoned_lbls = torch.cat([clean_train_lbls, poison_labels])

poisoned_dataset = TensorDataset(poisoned_imgs, poisoned_lbls)
poisoned_loader  = DataLoader(poisoned_dataset, batch_size=64, shuffle=True)

print(f"  Poisoned training set size: {len(poisoned_dataset)}")

# Train a brand-new model from scratch (same architecture + hyper-params)
poisoned_model = SimpleCNN(num_classes=4).to(device)
optimizer_p    = optim.Adam(poisoned_model.parameters(), lr=1e-3)

for ep in range(EPOCHS):
    loss = train_epoch(poisoned_model, poisoned_loader, optimizer_p, criterion)
    if (ep + 1) % 5 == 0 or ep == 0:
        print(f"  Epoch {ep+1:2d}/{EPOCHS}  loss={loss:.4f}")

print("\n  Evaluating poisoned model on test set:")
poisoned_acc, poisoned_per_class_correct, poisoned_per_class_total = \
    evaluate(poisoned_model, test_loader)

# Attack success: are target images now classified as BASE class?
poisoned_model.eval()
with torch.no_grad():
    target_preds_poisoned = poisoned_model(targets_dev).argmax(dim=1).cpu().numpy()

success_rate = float(np.mean(target_preds_poisoned == BASE_CLASS)) * 100.0
print(f"\n  Target preds (poisoned model)  : {target_preds_poisoned}")
print(f"  Attack Success Rate            : {success_rate:.2f}%")
print(f"  (Targets classified as class {BASE_CLASS} = {CLASS_NAMES[BASE_CLASS]})")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save Results
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Saving results and figures...")

with open('results.txt', 'w') as f:
    f.write(f"=== CS 228 Assignment 2 Results ===\n\n")
    f.write(f"Target Class : {TARGET_CLASS} ({CLASS_NAMES[TARGET_CLASS]})\n")
    f.write(f"Base Class   : {BASE_CLASS} ({CLASS_NAMES[BASE_CLASS]})\n")
    f.write(f"Num Poisons  : {NUM_POISONS}\n\n")
    f.write(f"Clean Model Accuracy  : {clean_acc:.2f}%\n")
    for i in range(4):
        if clean_per_class_total[i]:
            f.write(f"  Class {i} ({CLASS_NAMES[i]}): "
                    f"{100*clean_per_class_correct[i]/clean_per_class_total[i]:.2f}%\n")
    f.write(f"\nPoisoned Model Accuracy : {poisoned_acc:.2f}%\n")
    for i in range(4):
        if poisoned_per_class_total[i]:
            f.write(f"  Class {i} ({CLASS_NAMES[i]}): "
                    f"{100*poisoned_per_class_correct[i]/poisoned_per_class_total[i]:.2f}%\n")
    f.write(f"\nTarget predictions (clean model)   : {target_preds_clean}\n")
    f.write(f"Target predictions (poisoned model): {target_preds_poisoned}\n")
    f.write(f"Attack Success Rate: {success_rate:.2f}%\n")

# ── Side-by-side comparison figure ──────────────────────────────────────────
fig_cmp, axes_cmp = plt.subplots(NUM_POISONS, 3,
                                  figsize=(9, 3 * NUM_POISONS))
if NUM_POISONS == 1:
    axes_cmp = axes_cmp[np.newaxis, :]

for i in range(NUM_POISONS):
    axes_cmp[i, 0].imshow(
        np.transpose(bases[i].numpy(), (1, 2, 0)).clip(0, 1))
    axes_cmp[i, 0].set_title(f"Base (label={BASE_CLASS})", fontsize=8)
    axes_cmp[i, 0].axis('off')

    axes_cmp[i, 1].imshow(
        np.transpose(poisons[i].numpy(), (1, 2, 0)).clip(0, 1))
    axes_cmp[i, 1].set_title(f"Poison (label={BASE_CLASS})", fontsize=8)
    axes_cmp[i, 1].axis('off')

    axes_cmp[i, 2].imshow(
        np.transpose(targets[i].numpy(), (1, 2, 0)).clip(0, 1))
    axes_cmp[i, 2].set_title(f"Target (label={TARGET_CLASS})", fontsize=8)
    axes_cmp[i, 2].axis('off')

fig_cmp.suptitle("Base | Poison | Target", fontsize=12)
fig_cmp.tight_layout()
fig_cmp.savefig("final_poisons.png", dpi=100, bbox_inches='tight')
plt.close(fig_cmp)
print("  Saved: final_poisons.png")

# ── Accuracy comparison bar chart ───────────────────────────────────────────
fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
x     = np.arange(4)
width = 0.35
clean_accs    = [100*clean_per_class_correct[i]/clean_per_class_total[i]
                 if clean_per_class_total[i] else 0 for i in range(4)]
poisoned_accs = [100*poisoned_per_class_correct[i]/poisoned_per_class_total[i]
                 if poisoned_per_class_total[i] else 0 for i in range(4)]

bars1 = ax_bar.bar(x - width/2, clean_accs,    width, label='Clean Model',    color='steelblue')
bars2 = ax_bar.bar(x + width/2, poisoned_accs, width, label='Poisoned Model', color='tomato')
ax_bar.set_xlabel('Class')
ax_bar.set_ylabel('Accuracy (%)')
ax_bar.set_title('Per-Class Accuracy: Clean vs. Poisoned Model')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f"{i}\n({CLASS_NAMES[i]})" for i in range(4)])
ax_bar.set_ylim(0, 110)
ax_bar.legend()
for b in bars1:
    ax_bar.annotate(f"{b.get_height():.1f}",
                    xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
for b in bars2:
    ax_bar.annotate(f"{b.get_height():.1f}",
                    xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7)
fig_bar.tight_layout()
fig_bar.savefig("accuracy_comparison.png", dpi=100, bbox_inches='tight')
plt.close(fig_bar)
print("  Saved: accuracy_comparison.png")

print("\n=== Summary ===")
print(f"Clean Model Accuracy  : {clean_acc:.2f}%")
print(f"Poisoned Model Accuracy : {poisoned_acc:.2f}%")
print(f"Attack Success Rate   : {success_rate:.2f}%")
print(f"Target predictions (clean)   : {target_preds_clean}")
print(f"Target predictions (poisoned): {target_preds_poisoned}")
print("\nDone! All outputs saved.")
