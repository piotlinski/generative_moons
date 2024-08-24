import torch

from src.models.components.vae import VAE


class CVAE(VAE):
    """Conditional Variational Autoencoder."""

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Pass data through the encoder to get latent space parameters (mean & log-variance)."""
        mean, logvar = self.encoder(torch.cat([x, y], dim=1))
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(torch.cat([z, y], dim=1))

        return x_hat, {"mean": mean, "logvar": logvar, "z": z}
