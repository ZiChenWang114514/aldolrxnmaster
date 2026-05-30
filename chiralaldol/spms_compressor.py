"""Conv2D autoencoder for compressing SPMS matrices to flat vectors."""

import numpy as np
import torch
import torch.nn as nn


class SPMSAutoencoder(nn.Module):
    """Conv2D autoencoder: (C, 10, 20) → latent_dim → (C, 10, 20)."""

    def __init__(self, n_channels=2, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (16, 5, 10)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # (32, 5, 5)
            nn.Flatten(),  # 32*5*5 = 800
            nn.Linear(800, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 800),
            nn.ReLU(),
            nn.Unflatten(1, (32, 5, 5)),
            nn.Upsample(size=(5, 10)),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(10, 20)),
            nn.Conv2d(16, n_channels, kernel_size=(3, 5), padding=(1, 2)),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(spms_arrays, n_channels=2, latent_dim=16,
                      epochs=100, lr=1e-3, batch_size=64):
    """Train autoencoder on SPMS matrices.

    Args:
        spms_arrays: (N, n_channels, 10, 20) float32 array
        n_channels: number of SPMS channels (2 for ketone+aldehyde)
        latent_dim: latent dimension per sample
        epochs: training epochs
        lr: learning rate
        batch_size: batch size

    Returns:
        (model, latent_vectors) where latent_vectors is (N, latent_dim)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(spms_arrays).float().to(device)

    # Normalize
    mean = X.mean()
    std = X.std() + 1e-8
    X_norm = (X - mean) / std

    model = SPMSAutoencoder(n_channels, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_norm)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(X)
            print(f"  AE epoch {epoch+1}/{epochs}: MSE={avg_loss:.6f}")

    # Extract latent vectors
    model.eval()
    with torch.no_grad():
        latent = model.encode(X_norm).cpu().numpy()

    return model, latent, float(mean), float(std)


def compress_spms_pca(spms_arrays, n_components=16):
    """Alternative: PCA compression of flattened SPMS.

    Args:
        spms_arrays: (N, n_channels, 10, 20)
        n_components: number of PCA components

    Returns:
        (N, n_components) compressed vectors
    """
    from sklearn.decomposition import PCA

    N = spms_arrays.shape[0]
    flat = spms_arrays.reshape(N, -1)  # (N, n_channels*200)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(flat), pca


def extract_spms_stats(spms_arrays, channel_names=("ket", "ald")):
    """Extract statistical features from SPMS arrays for tree models.

    Args:
        spms_arrays: (N, n_channels, n_theta, n_phi) float32
        channel_names: names for each channel

    Returns:
        (features_array, feature_names) where features_array is (N, n_channels*5)
    """
    feats = []
    names = []
    for ch, ch_name in enumerate(channel_names):
        if ch >= spms_arrays.shape[1]:
            break
        arr = spms_arrays[:, ch]  # (N, n_theta, n_phi)
        n_theta = arr.shape[1]
        half = n_theta // 2
        feats.extend([
            arr.mean(axis=(1, 2)),
            arr.std(axis=(1, 2)),
            arr.min(axis=(1, 2)),
            arr.max(axis=(1, 2)),
            arr[:, :half, :].mean(axis=(1, 2)) - arr[:, half:, :].mean(axis=(1, 2)),
        ])
        names.extend([
            f"spms_{ch_name}_mean", f"spms_{ch_name}_std",
            f"spms_{ch_name}_min", f"spms_{ch_name}_max",
            f"spms_{ch_name}_asymm",
        ])
    return np.column_stack(feats).astype(np.float32), names
