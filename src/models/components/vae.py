import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder network for VAE."""

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | int = 8, latent_dim: int = 4):
        """
        :param input_dim: number of input features
        :param hidden_dims: list of hidden layer dimensions (or single hidden layer dimension)
        :param latent_dim: latent space dimension
        """
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        hidden_dim = hidden_dims[0]
        encoder = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for dim in hidden_dims[1:]:
            encoder.append(nn.Linear(hidden_dim, dim))
            encoder.append(nn.ReLU(inplace=True))
            hidden_dim = dim

        self.encoder = nn.Sequential(*encoder)
        self.lin_mean = nn.Linear(hidden_dim, latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass data through the encoder to get latent space parameters (mean & log-variance)."""
        x = self.encoder(x)

        mean = self.lin_mean(x)
        logvar = self.lin_logvar(x)

        return mean, logvar


class Decoder(nn.Module):
    """Decoder network for VAE."""

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | int = 8, output_dim: int = 2):
        """
        :param latent_dim: latent space dimension
        :param hidden_dims: list of hidden layer dimensions (or single hidden layer dimension)
        :param output_dim: number of output features
        """
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        hidden_dim = hidden_dims[0]
        decoder = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for dim in hidden_dims[1:]:
            decoder.append(nn.Linear(hidden_dim, dim))
            decoder.append(nn.ReLU(inplace=True))
            hidden_dim = dim

        decoder.append(nn.Linear(hidden_dim, output_dim))

        self.decoder = nn.Sequential(*decoder)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Pass data through the decoder to get reconstructed output."""
        return self.decoder(z)


class VAE(nn.Module):
    """Variational Autoencoder."""

    def __init__(self, encoder: Encoder, decoder: Decoder, eps_w: float = 1.0):
        """
        :param encoder: encoder network
        :param decoder: decoder network
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.eps_w = eps_w

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick to sample from the latent space.

        .. note: we use the log-variance instead of the variance for numerical stability.
        """
        std = torch.exp(0.5 * logvar)
        eps = self.eps_w * torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Perform a forward pass through the VAE.

        .. note: we ignore the conditional tensor `y` in the base VAE.
        """
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)

        return x_hat, {"mean": mean, "logvar": logvar, "z": z}


class VAELoss(nn.Module):
    """Loss function for VAE.

    .. note: we use a negative ELBO loss.
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        """
        :param beta: weight for the KL divergence term
        :param reduction: reduction method for the loss (mean, sum)
        """
        super().__init__()

        self.beta = beta
        self.reduction = reduction

    def forward(
        self, predictions: tuple[torch.Tensor, dict[str, torch.Tensor]], x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate the negative ELBO loss.

        .. note: we use a closed-form KL divergence for Gaussian distributions
            KL(q || N(0, I)) = 0.5 * sum(sigma^2 + mu^2 - log(sigma^2) - 1)
        """
        x_hat, zs = predictions

        recon_error = F.mse_loss(x_hat, x, reduction="sum")
        kl_div = -0.5 * torch.sum(1 + zs["logvar"] - zs["mean"].pow(2) - zs["logvar"].exp())

        loss = recon_error + self.beta * kl_div

        if self.reduction == "mean":
            loss /= x.numel()

        return loss, {"reconstruction_error": recon_error, "kl_divergence": kl_div}
