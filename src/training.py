"""
Training: the free rotation-augmentation trick, and the training loop itself.

augment_rotate:
    Rotating a spherical signal about the polar axis by angle `alpha` is
    exactly a per-coefficient phase multiply:

        f_hat[l, m]  ->  f_hat[l, m] * exp(i * m * alpha)

    for rotation angle `alpha`. No resampling, no interpolation error, and
    it's differentiable and batched trivially on the GPU -- so it's applied
    fresh to every training batch as cheap, exact data augmentation.

train:
    Mini-batch Adam with weight decay, a cosine learning-rate schedule, the
    rotation augmentation above applied each batch, and best-validation-
    checkpoint tracking/restoration (rather than just reporting whatever the
    final epoch happened to land on).
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F


def augment_rotate(x, m_orders):
    """Apply a random polar-axis rotation to each sample in a batch.

    Args:
        x: (N, C, n_harm) complex tensor of harmonic coefficients.
        m_orders: (n_harm,) tensor giving the order m of each coefficient
            (same for every sample -- this is `SHT.lm_order`).

    Returns:
        (N, C, n_harm) complex tensor: `x` rotated by an independent random
        angle alpha for each of the N samples in the batch.
    """
    # One random rotation angle per sample in the batch.
    alpha = torch.rand(x.shape[0], device=x.device) * 2 * np.pi

    # exp(i * m * alpha), broadcast to (N, n_harm): each coefficient's phase
    # advances by its own order m times that sample's rotation angle.
    phase = torch.exp(1j * m_orders.view(1, -1) * alpha.view(-1, 1))

    # Broadcast over the channel dimension C and multiply elementwise.
    return x * phase.unsqueeze(1)


def train(model, sht, X_train, y_train, X_val, y_val,
          epochs=40, batch_size=32, lr=2e-3, weight_decay=1e-4, augment=True,
          device="cuda"):
    """Train `model` in place and return the best validation accuracy seen.

    Notably, at the end of training the model's weights are reset to
    whichever epoch had the best validation accuracy -- not just whatever
    the final epoch happened to land on, which can be worse due to
    late-training oscillation or overfitting.

    Args:
        model: a `SphericalCNN` instance.
        sht: the `SHT` instance the model was built with (needed for
            `lm_degree`/`lm_order`, used by the conv layers and augmentation).
        X_train, y_train, X_val, y_val: train/validation tensors.
        epochs: number of passes over the training set.
        batch_size: mini-batch size.
        lr: initial learning rate (Adam).
        weight_decay: L2 regularization strength (Adam).
        augment: whether to apply free rotation augmentation to each
            training batch.
        device: device to run training on.

    Returns:
        float: the best validation accuracy observed across all epochs.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = X_train.shape[0]
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()

        # Shuffle the training set each epoch.
        perm = torch.randperm(n)
        X_train, y_train = X_train[perm], y_train[perm]
        epoch_loss = 0.0

        for b in range(0, n, batch_size):
            xb = X_train[b:b + batch_size].to(device)
            yb = y_train[b:b + batch_size].to(device)

            if augment:
                # Cheap, exact rotation augmentation done in harmonic space.
                xb = augment_rotate(xb, sht.lm_order)

            opt.zero_grad()
            logits = model(xb, sht.lm_degree)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)

        scheduler.step()
        epoch_loss /= n

        # Evaluate on a train subset (cheap proxy for train accuracy) and
        # the full validation set.
        model.eval()
        with torch.no_grad():
            train_acc = (
                model(X_train[:300].to(device), sht.lm_degree).argmax(1).cpu()
                == y_train[:300]
            ).float().mean().item()
            val_acc = (
                model(X_val.to(device), sht.lm_degree).argmax(1).cpu()
                == y_val
            ).float().mean().item()

        # Keep a snapshot of the model whenever validation accuracy improves.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"epoch {epoch:2d}/{epochs}  loss={epoch_loss:.4f}  "
              f"train_acc~={train_acc:.3f}  val_acc={val_acc:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.5f}")

    # Restore the best-performing checkpoint rather than keeping the last epoch.
    model.load_state_dict(best_state)
    print(f"\nBest validation accuracy (restored): {best_val_acc:.3f}")
    return best_val_acc
