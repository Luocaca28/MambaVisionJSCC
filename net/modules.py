import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_

try:
    from einops import rearrange, repeat
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'einops' required for MambaVisionMixer. "
        "Install via `pip install einops`."
    ) from e

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'mamba-ssm' required for MambaVisionMixer. "
        "Install via `pip install mamba-ssm`."
    ) from e


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, -1)
    )
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_gate: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
            )
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gate = bool(attn_gate)
        if self.attn_gate:
            self.gate_proj = nn.Linear(dim, dim)
            self.gate_act = nn.Sigmoid()
        else:
            self.gate_proj = None
            self.gate_act = None

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        x_in = x

        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x_attn = attn @ v
        if self.attn_gate:
            gate = self.gate_act(self.gate_proj(x_in))
            gate = gate.view(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            x_attn = x_attn * gate
        x = x_attn.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_gate: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_gate=attn_gate,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.register_buffer("attn_mask", self._build_attn_mask(device=None))

    def _build_attn_mask(self, device: torch.device | None) -> torch.Tensor | None:
        if self.shift_size <= 0:
            return None
        H, W = self.input_resolution
        if device is None:
            device = torch.device("cpu")
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        return attn_mask

    def update_mask(self, device: torch.device | None = None):
        self.attn_mask = self._build_attn_mask(device=device)

    def forward(
        self, x: torch.Tensor, resolution: tuple[int, int] | None = None
    ) -> torch.Tensor:
        if resolution is not None and tuple(resolution) != tuple(self.input_resolution):
            self.input_resolution = tuple(resolution)
            self.update_mask(device=x.device)

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(
        self,
        input_resolution,
        dim: int,
        out_dim: int = None,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        out_dim = dim if out_dim is None else out_dim
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H * W // 4, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchReverseMerging(nn.Module):
    def __init__(
        self,
        input_resolution,
        dim: int,
        out_dim: int,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        x = self.norm(x)
        x = self.increment(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = nn.PixelShuffle(2)(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape

        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())

        x = F.silu(
            F.conv1d(
                input=x,
                weight=self.conv1d_x.weight,
                bias=self.conv1d_x.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )
        z = F.silu(
            F.conv1d(
                input=z,
                weight=self.conv1d_z.weight,
                bias=self.conv1d_z.bias,
                padding="same",
                groups=self.d_inner // 2,
            )
        )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        attn_gate: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gate = bool(attn_gate)
        if self.attn_gate:
            self.gate_proj = nn.Linear(dim, dim)
            self.gate_act = nn.Sigmoid()
        else:
            self.gate_proj = None
            self.gate_act = None

    def forward(self, x):
        B, N, C = x.shape
        x_in = x
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        if self.attn_gate:
            gate = self.gate_act(self.gate_proj(x_in))
            gate = gate.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            x = x * gate
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


MVAttention = Attention


class MambaVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        num_heads: int = 4,
        use_mamba: bool = True,
        use_attention: bool = False,
        window_size: int = 0,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 3,
        mamba_expand: int = 1,
        attn_gate: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.use_attention = bool(use_attention)
        self.use_mamba = bool(use_mamba) and not self.use_attention
        self.window_size = int(window_size) if window_size is not None else 0
        self.window_pad_mode = "reflect"

        self.norm1 = nn.LayerNorm(dim)
        if self.use_attention:
            self.mixer = MVAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=False,
                attn_drop=0.0,
                proj_drop=0.0,
                norm_layer=nn.LayerNorm,
                attn_gate=attn_gate,
            )
        elif self.use_mamba:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=max(1, int(mamba_d_state)),
                d_conv=max(1, int(mamba_d_conv)),
                expand=max(1, int(mamba_expand)),
            )
        else:
            self.mixer = nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0 else nn.Identity()
        use_layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        if use_layer_scale:
            ls = float(layer_scale)
            self.gamma_1 = nn.Parameter(ls * torch.ones(dim))
            self.gamma_2 = nn.Parameter(ls * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def _window_partition(
        self, x: torch.Tensor, H: int, W: int
    ) -> tuple[torch.Tensor, tuple[int, int, int, int, int, int]]:
        B, _, C = x.shape
        ws = int(self.window_size)
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        Hp = H + pad_b
        Wp = W + pad_r

        x_2d = x.view(B, H, W, C)
        if pad_r > 0 or pad_b > 0:
            pad_mode = self.window_pad_mode
            if pad_mode == "reflect" and (pad_r >= W or pad_b >= H or W <= 1 or H <= 1):
                pad_mode = "replicate"
            x_2d = F.pad(x_2d, (0, 0, 0, pad_r, 0, pad_b), mode=pad_mode)

        windows = (
            x_2d.view(B, Hp // ws, ws, Wp // ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, ws * ws, C)
        )
        return windows, (H, W, Hp, Wp, pad_b, pad_r)

    def _window_reverse(self, windows: torch.Tensor, meta: tuple[int, int, int, int, int, int]) -> torch.Tensor:
        H, W, Hp, Wp, pad_b, pad_r = meta
        ws = int(self.window_size)
        num_windows_per_img = (Hp // ws) * (Wp // ws)
        B = windows.shape[0] // num_windows_per_img

        x_2d = (
            windows.view(B, Hp // ws, Wp // ws, ws, ws, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, Hp, Wp, -1)
        )
        if pad_b > 0 or pad_r > 0:
            x_2d = x_2d[:, :H, :W, :].contiguous()
        return x_2d.view(B, H * W, -1)

    def _apply_mixer(self, x: torch.Tensor, resolution: tuple[int, int] | None) -> torch.Tensor:
        if self.use_mamba:
            return self.mixer(x)
        if self.window_size <= 0 or resolution is None:
            return self.mixer(x)
        H, W = resolution
        if H <= 0 or W <= 0 or x.shape[1] != H * W:
            return self.mixer(x)

        windows, meta = self._window_partition(x, H, W)
        windows = self.mixer(windows)
        return self._window_reverse(windows, meta)

    def forward(self, x: torch.Tensor, resolution: tuple[int, int] | None = None) -> torch.Tensor:
        y = self._apply_mixer(self.norm1(x), resolution)
        if self.gamma_1 is not None:
            y = y * self.gamma_1
        x = x + self.drop_path(y)

        y = self.mlp(self.norm2(x))
        if self.gamma_2 is not None:
            y = y * self.gamma_2
        x = x + self.drop_path(y)
        return x


def _expand_drop_path(drop_path, depth: int) -> list[float]:
    if depth <= 0:
        return []
    if drop_path is None:
        return [0.0 for _ in range(depth)]
    if isinstance(drop_path, (list, tuple)):
        if len(drop_path) != depth:
            raise ValueError(f"drop_path length mismatch: got {len(drop_path)}, expected {depth}.")
        return [float(x) for x in drop_path]
    return [float(drop_path) for _ in range(depth)]


class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int = 4,
        stage_index: int = 0,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        window_size: int = 0,
        drop_path: float | list[float] | tuple[float, ...] = 0.0,
        layer_scale: float | None = None,
        layer_scale_conv: float | None = None,
        conv_kernel_size: int = 3,
        conv_token_mlp_ratio: float = 0.0,
        downsample=None,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 3,
        mamba_expand: int = 1,
        attn_gate_swin: bool = False,
        attn_gate_mv: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.downsample = (
            downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
            if downsample is not None
            else None
        )

        block_dim = out_dim
        dprs = _expand_drop_path(drop_path, depth)
        block_resolution = input_resolution
        if self.downsample is not None:
            block_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)

        if stage_index <= 1:
            self.blocks = nn.ModuleList(
                [
                    SwinTransformerBlock(
                        dim=block_dim,
                        input_resolution=block_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        qk_scale=None,
                        norm_layer=norm_layer,
                        attn_gate=attn_gate_swin,
                    )
                    for i in range(depth)
                ]
            )
        else:
            mamba_depth = depth if depth <= 1 else max(1, min(depth // 2, depth - 1))
            self.blocks = nn.ModuleList()
            for i in range(depth):
                if i < mamba_depth:
                    self.blocks.append(
                        MambaVisionBlock(
                            dim=block_dim,
                            mlp_ratio=mlp_ratio,
                            num_heads=num_heads,
                            use_mamba=True,
                            use_attention=False,
                            window_size=0,
                            drop_path=dprs[i],
                            layer_scale=layer_scale,
                            mamba_d_state=mamba_d_state,
                            mamba_d_conv=mamba_d_conv,
                            mamba_expand=mamba_expand,
                            attn_gate=attn_gate_mv,
                        )
                    )
                else:
                    self.blocks.append(
                        SwinTransformerBlock(
                            dim=block_dim,
                            input_resolution=block_resolution,
                            num_heads=num_heads,
                            window_size=window_size,
                            shift_size=0 if (i % 2 == 0) else window_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            norm_layer=norm_layer,
                            attn_gate=attn_gate_swin,
                        )
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        if self.downsample is not None:
            x = self.downsample(x)
            H, W = H // 2, W // 2
        for blk in self.blocks:
            x = blk(x, (H, W))
        return x

    def update_resolution(self, H: int, W: int):
        self.input_resolution = (H, W)
        block_h, block_w = H, W
        if self.downsample is not None:
            block_h, block_w = H // 2, W // 2
        for blk in self.blocks:
            if hasattr(blk, "input_resolution"):
                blk.input_resolution = (block_h, block_w)
            if hasattr(blk, "update_mask"):
                blk.update_mask(device=next(self.parameters()).device)
        if self.downsample is not None:
            self.downsample.input_resolution = (H, W)


class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int = 4,
        stage_index: int = 0,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        window_size: int = 0,
        drop_path: float | list[float] | tuple[float, ...] = 0.0,
        layer_scale: float | None = None,
        layer_scale_conv: float | None = None,
        conv_kernel_size: int = 3,
        conv_token_mlp_ratio: float = 0.0,
        upsample=None,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 3,
        mamba_expand: int = 1,
        attn_gate_swin: bool = False,
        attn_gate_mv: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        self.depth = depth
        dprs = _expand_drop_path(drop_path, depth)

        if stage_index <= 1:
            mamba_depth = depth if depth <= 1 else max(1, min(depth // 2, depth - 1))
            self.blocks = nn.ModuleList()
            for i in range(depth):
                if i < mamba_depth:
                    self.blocks.append(
                        MambaVisionBlock(
                            dim=dim,
                            mlp_ratio=mlp_ratio,
                            num_heads=num_heads,
                            use_mamba=True,
                            use_attention=False,
                            window_size=0,
                            drop_path=dprs[i],
                            layer_scale=layer_scale,
                            mamba_d_state=mamba_d_state,
                            mamba_d_conv=mamba_d_conv,
                            mamba_expand=mamba_expand,
                            attn_gate=attn_gate_mv,
                        )
                    )
                else:
                    self.blocks.append(
                        SwinTransformerBlock(
                            dim=dim,
                            input_resolution=input_resolution,
                            num_heads=num_heads,
                            window_size=window_size,
                            shift_size=0 if (i % 2 == 0) else window_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True,
                            qk_scale=None,
                            norm_layer=norm_layer,
                            attn_gate=attn_gate_swin,
                        )
                    )
        else:
            self.blocks = nn.ModuleList(
                [
                    SwinTransformerBlock(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        qk_scale=None,
                        norm_layer=norm_layer,
                        attn_gate=attn_gate_swin,
                    )
                    for i in range(depth)
                ]
            )

        self.upsample = (
            upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
            if upsample is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        for blk in self.blocks:
            x = blk(x, (H, W))
        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def update_resolution(self, H: int, W: int):
        self.input_resolution = (H, W)
        for blk in self.blocks:
            if hasattr(blk, "input_resolution"):
                blk.input_resolution = (H, W)
            if hasattr(blk, "update_mask"):
                blk.update_mask(device=next(self.parameters()).device)
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)
