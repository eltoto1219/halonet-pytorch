import torch

from halonet_pytorch import HaloAttention

attn = HaloAttention(
    dim=512,  # dimension of feature map
    # neighborhood block size (feature map must be divisible by this)
    block_size=8,
    halo_size=4,  # halo size (block receptive field)
    dim_head=64,  # dimension of each head
    heads=4,  # number of attention heads
)

fmap = torch.randn(1, 512, 32, 32)
fmap_out = attn(fmap)  # (1, 512, 32, 32)
