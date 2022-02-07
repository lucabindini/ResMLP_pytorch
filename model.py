import torch
from torch import nn
from einops.layers.torch import Rearrange


class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        return self.alpha * x + self.beta

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class MLPblock(nn.Module):
    def __init__(self, dim, num_patches, layerscale_init=1e-4):
        super().__init__()
        self.pre_affine = Aff(dim)
        self.token_mix = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patches, num_patches),
            Rearrange('b d n -> b n d'),
        )
        self.ff = nn.Sequential(
            FeedForward(dim, 4*dim),
        )
        self.post_affine = Aff(dim)
        self.layerscale_1 = nn.Parameter(layerscale_init * torch.ones((dim)), requires_grad=True)
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.layerscale_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.layerscale_2 * self.ff(x)
        return x


class ResMLP(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth):
        super().__init__()
        assert image_size % patch_size == 0, 'dim must be divisible by patch size'
        self.num_patches =  (image_size// patch_size) ** 2
        self.embeddings = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mlp_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(dim, self.num_patches))
        self.affine = Aff(dim)
        self.linear_classifier = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.embeddings(x)
        for blk in self.mlp_blocks:
            x = blk(x)
        x = self.affine(x)
        x = x.mean(dim=1)
        return self.linear_classifier(x)