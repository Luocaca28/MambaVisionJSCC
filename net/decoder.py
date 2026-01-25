import torch
import torch.nn as nn

from net.modules import PatchReverseMerging, trunc_normal_
from net.modules import MambaDecoderLayer
from net.encoder import AdaptiveModulator


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU(approximate="tanh")
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class MambaJSCC_Decoder(nn.Module):
    """
    MambaVision-based decoder for JSCC.

    结构与 SwinJSCC_Decoder 对齐：
      - 多个 stage（MambaDecoderLayer）
      - 可选 SNR-aware 解码调制
      - 输出尺寸与输入图像一致 (B,3,H,W)
    """

    def __init__(
        self,
        img_size,
        embed_dims,
        depths,
        num_heads,
        C,
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        drop_path_rate: float = 0.0,
        layer_scale: float | None = None,
        layer_scale_conv: float | None = None,
        conv_kernel_size: int = 3,
        conv_token_mlp_ratio: float = 0.0,
        highres_mixer: str = "conv",
        post_upsample_attn_depth: int = 0,
        refine_head: bool = True,
        refine_channels: int = 32,
        refine_depth: int = 2,
        refine_scale: float = 0.1,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        attn_gate: bool = False,
        attn_gate_swin: bool | None = None,
        attn_gate_mv: bool | None = None,
        bottleneck_dim=16,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (
            img_size[0] // 2 ** len(depths),
            img_size[1] // 2 ** len(depths),
        )
        num_patches = self.H // 4 * self.W // 4
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dims[0])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        # hierarchical decoder stages
        if layer_scale_conv is None:
            layer_scale_conv = layer_scale

        total_depth = int(sum(depths))
        if float(drop_path_rate) > 0 and total_depth > 0:
            dpr = torch.linspace(0, float(drop_path_rate), total_depth).tolist()
        else:
            dpr = [0.0 for _ in range(total_depth)]
        dpr_ptr = 0
        if attn_gate_swin is None:
            attn_gate_swin = bool(attn_gate)
        if attn_gate_mv is None:
            attn_gate_mv = bool(attn_gate)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MambaDecoderLayer(
                dim=int(embed_dims[i_layer]),
                out_dim=int(embed_dims[i_layer + 1])
                if (i_layer < self.num_layers - 1)
                else 3,
                input_resolution=(
                    self.patches_resolution[0] * (2 ** i_layer),
                    self.patches_resolution[1] * (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=self.num_heads[i_layer]
                if isinstance(self.num_heads, (list, tuple))
                else int(self.num_heads),
                stage_index=i_layer,
                mlp_ratio=self.mlp_ratio,
                norm_layer=norm_layer,
                window_size=window_size,
                drop_path=dpr[dpr_ptr : dpr_ptr + int(depths[i_layer])],
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                conv_kernel_size=conv_kernel_size,
                conv_token_mlp_ratio=conv_token_mlp_ratio,
                highres_mixer=highres_mixer,
                post_upsample_attn_depth=post_upsample_attn_depth,
                upsample=PatchReverseMerging,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                attn_gate_swin=attn_gate_swin,
                attn_gate_mv=attn_gate_mv,
            )
            dpr_ptr += int(depths[i_layer])
            self.layers.append(layer)

        if C is not None:
            self.head_list = nn.Linear(C, embed_dims[0])

        self.refine_scale = float(refine_scale)
        if bool(refine_head):
            ch = max(8, int(refine_channels))
            depth = max(1, int(refine_depth))
            blocks = [ResidualRefineBlock(ch) for _ in range(depth)]
            self.refine_head = nn.Sequential(
                nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1),
                nn.GELU(approximate="tanh"),
                *blocks,
                nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.refine_head = None
        self.apply(self._init_weights)

        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dims[0], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[0]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, snr, model):
        if model == "MambaVisionJSCC_w/o_SAandRA":
            x = self.head_list(x)
            for layer in self.layers:
                x = layer(x)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return self._refine(x)

        elif model == "MambaVisionJSCC_w/_SA":
            B, L, C = x.size()
            device = x.device
            x = self.head_list(x)
            snr_cuda = torch.tensor(snr, dtype=torch.float, device=device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)
                bm = (
                    self.bm_list[i](snr_batch)
                    .unsqueeze(1)
                    .expand(-1, L, -1)
                )
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            for layer in self.layers:
                x = layer(x)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return self._refine(x)

        raise ValueError(f"Unsupported model variant (RA removed): {model}")

    def _refine(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        if self.refine_head is None:
            return x
        residual = self.refine_head(x) * self.refine_scale
        return (x + residual).clamp(0.0, 1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer), W * (2 ** i_layer))


def create_mamba_decoder(**kwargs):
    return MambaJSCC_Decoder(**kwargs)
