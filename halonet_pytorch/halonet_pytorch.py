import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

# relative positional embedding


def to(x):
    return {"device": x.device, "dtype": x.dtype}


def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, "b l c -> b (l c)")
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum("b x y d, r d -> b x y r", q, rel_k)
    logits = rearrange(logits, "b x y r -> (b x) y r")
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, "b (x y) c -> b x y c", x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b x i y j-> b (x y) (i j)")

        q = rearrange(q, "b x y d -> b y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b x i y j -> b (y x) (j i)")
        return rel_logits_w + rel_logits_h


# classes


class HaloAttention(nn.Module):
    def __init__(self, *, dim, block_size, halo_size, dim_head=64, heads=8):
        super().__init__()
        assert halo_size > 0, "halo size must be greater than 0"

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size=block_size,
            rel_size=block_size + (halo_size * 2),
            dim_head=dim_head,
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, c, h, w, block, halo, heads, device = (
            *x.shape,
            self.block_size,
            self.halo_size,
            self.heads,
            x.device,
        )
        assert h % block == 0 and w % block == 0, (
            "fmap dimensions must be divisible by the block size",
            h,
            w,
            block,
        )
        assert (
            c == self.dim
        ), f"channels for input ({c}) does not equal to the correct dimension ({self.dim})"

        # get block neighborhoods, and prepare a halo-ed version (blocks with padding) for deriving key values
        # block is 8, halo is 4

        q_inp = rearrange(
            x, "b c (h p1) (w p2) -> (b h w) (p1 p2) c", p1=block, p2=block
        )

        # hello

        kv_inp = F.unfold(x, kernel_size=block + halo * 2, stride=block, padding=halo)
        # x (1, 512, 32, 32)
        kv_inp = rearrange(kv_inp, "b (c j) i -> (b i) j c", c=c)

        # kv_inp --> this is the halo'ed part of the image
        # derive queries, keys, values

        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim=-1)

        # split heads

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=heads), (q, k, v)
        )

        # scale

        q *= self.scale

        # attention

        sim = einsum("b i d, b j d -> b i j", q, k)

        # add relative positional bias
        pos_bias = self.rel_pos_emb(q)

        sim += pos_bias

        # mask out padding (in the paper, they claim to not need masks, but what about padding?)

        mask = torch.ones(1, 1, h, w, device=device)
        mask = F.unfold(
            mask, kernel_size=block + (halo * 2), stride=block, padding=halo
        )
        mask = repeat(mask, "() j i -> (b i h) () j", b=b, h=heads)
        mask = mask.bool()

        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1)

        # we have attention shape:
        # attn shape, v shape
        # torch.Size([64, 64, 256]), torch.Size([64, 256, 64]))
        # out shape after rearagne
        # torch.Size([64, 64, 64]) --> torch.Size([16, 64, 256])

        # aggregate

        out = einsum("b i j, b j d -> b i d", attn, v)

        # merge and combine heads

        out = rearrange(out, "(b h) n d -> b n (h d)", h=heads)

        out = self.to_out(out)

        # merge blocks back to original feature map

        out = rearrange(
            out,
            "(b h w) (p1 p2) c -> b c (h p1) (w p2)",
            b=b,
            h=(h // block),
            w=(w // block),
            p1=block,
            p2=block,
        )
        return out


# HALO STACK


class HaloBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out,
        proj_factor,
        downsample,
        heads=4,
        dim_head=128,
        rel_pos_emb=False,
        activation=nn.ReLU(),
    ):
        super().__init__()

        # shortcut

        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    dim,
                    dim_out,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(dim_out),
                activation,
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion

        attn_dim_in = dim_out // proj_factor
        attn_dim_out = heads * dim_head

        self.net = nn.Sequential(
            nn.Conv2d(dim, attn_dim_in, 1, bias=False),
            nn.BatchNorm2d(attn_dim_in),
            activation,
            HaloAttention(
                dim=attn_dim_in,
                # fmap_size=fmap_size,
                block_size=7,
                halo_size=7,
                heads=heads,
                dim_head=dim_head,
                # rel_pos_emb=rel_pos_emb,
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
        )

        # init last batch norm gamma to zero

        nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x = x + shortcut
        return self.activation(x)


class HaloStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out=2048,
        proj_factor=4,
        num_layers=3,
        heads=4,
        dim_head=128,
        downsample=True,
        rel_pos_emb=False,
        activation=nn.ReLU(),
    ):
        super().__init__()
        fmap_size = pair(fmap_size)

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = dim if is_first else dim_out
            layer_downsample = is_first and downsample

            fmap_divisor = 2 if downsample and not is_first else 1
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(
                HaloBlock(
                    dim=dim,
                    fmap_size=layer_fmap_size,
                    dim_out=dim_out,
                    proj_factor=proj_factor,
                    heads=heads,
                    dim_head=dim_head,
                    downsample=layer_downsample,
                    rel_pos_emb=rel_pos_emb,
                    activation=activation,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert (
            c == self.dim
        ), f"channels of feature map {c} must match channels given at init {self.dim}"
        assert (
            h == self.fmap_size[0] and w == self.fmap_size[1]
        ), f"height and width ({h} {w}) of feature map must match the fmap_size given at init {self.fmap_size}"
        return self.net(x)
