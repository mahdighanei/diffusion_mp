import torch
import torch.nn as nn
import torch.nn.functional as F



class AENet3(nn.Module):
    """Simple auto-encoder model for point-clouds
    """
    def __init__(self, hparams):
        super().__init__()
        """
        Arguments:
            latent_size: an integer, dimension of the representation vector.
            num_points: an integer.
        """

        latent_size = hparams['ae_latentsize']
        dim = 3
        num_points = hparams['ae_num_ptclouds']
        self.sample_input = torch.randn((32, dim, num_points))

        # ENCODER
        pointwise_layers = []
        num_units = [dim, 64, 128, 128, 256, 512, latent_size]

        for i in range(1, len(num_units)):
            n = num_units[i - 1]
            m = num_units[i]
            pointwise_layers.extend(
                [
                    nn.Conv1d(n, m, kernel_size=1, bias=False),
                    nn.BatchNorm1d(m),
                    nn.ReLU(inplace=True),
                ]
            )

        self.pointwise_layers = nn.Sequential(*pointwise_layers)
        self.pooling = nn.AdaptiveMaxPool1d(1)

        # DECODER
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_size, 256, kernel_size=1, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=1, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, dim * num_points, kernel_size=1),
        )
    
        numparams = sum(p.numel() for p in self.pointwise_layers.parameters() if p.requires_grad)
        print('encoder params:', numparams)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            encoding: a float tensor with shape [b, latent_size].
            restoration: a float tensor with shape [b, 3, num_points].
        """
        x = x['pt_clouds']
        b, dim, num_points = x.size()
        x = self.pointwise_layers(x)  # shape [b, latent_size, num_points]
        encoding = self.pooling(x)  # shape [b, latent_size, 1]
        if self.decoder is None:
            return encoding, None

        x = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        restoration = x.view(b, dim, num_points)

        return encoding, restoration