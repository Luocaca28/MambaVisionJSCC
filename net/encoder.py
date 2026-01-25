import torch
import torch.nn as nn

from net.modules import PatchEmbed, PatchMerging, trunc_normal_, MambaEncoderLayer


class AdaptiveModulator(nn.Module):
    """Reimplementation of the SwinJSCC AdaptiveModulator."""

    def __init__(self, M: int):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MambaJSCC_Encoder(nn.Module):
    """
    MambaVision-based encoder for JSCC.

    接口基本仿照 SwinJSCC_Encoder：
      - 输入: (B, 3, H, W)
      - 输出:
          * 无 RA: feature (B, L, C)
          * 有 RA: (feature, mask)，其中 mask (B, L, C)
    """

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dims,
        depths,
        num_heads,
        C,
        window_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        drop_path_rate: float = 0.0,
        layer_scale: float | None = None,
        layer_scale_conv: float | None = None,
        conv_kernel_size: int = 3,
        conv_token_mlp_ratio: float = 0.0,
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
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        if isinstance(patch_size, (list, tuple)):
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        else:
            patch_h = patch_w = int(patch_size)
        self.patches_resolution = (img_size[0] // patch_h, img_size[1] // patch_w)
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])

        self.hidden_dim = int(self.embed_dims[-1] * 1.5)
        self.layer_num = layer_num = 7

        # hierarchical MambaVision encoder stages
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
            # PatchEmbed downsamples by `patch_size`, then each stage (except stage 0) downsamples by 2 via PatchMerging.
            # In this implementation, PatchMerging is applied at the *start* of stage i_layer>=1, so:
            #   - stage0 operates at patches_resolution (H/patch, W/patch)
            #   - stage1 downsamples to patches_resolution/2
            #   - stage2 downsamples to patches_resolution/4, ...
            if i_layer <= 1:
                input_resolution = self.patches_resolution
            else:
                div = 2 ** (i_layer - 1)
                input_resolution = (
                    self.patches_resolution[0] // div,
                    self.patches_resolution[1] // div,
                )
            layer = MambaEncoderLayer(
                dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else int(embed_dims[0]),
                out_dim=int(embed_dims[i_layer]),
                input_resolution=input_resolution,
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
                downsample=PatchMerging if i_layer != 0 else None,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                attn_gate_swin=attn_gate_swin,
                attn_gate_mv=attn_gate_mv,
            )
            dpr_ptr += int(depths[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(embed_dims[-1])
        if C is not None:
            self.head_list = nn.Linear(embed_dims[-1], C)

        self.apply(self._init_weights)

        # SA / RA 调制模块
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dims[-1], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[-1]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

        # SNR+Rate 组合调制
        # (RA removed) no rate-aware modulation branch.

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Backward-compat: older checkpoints used *_list1 naming for the SA branch.
        legacy_map = (
            ("bm_list1.", "bm_list."),
            ("sm_list1.", "sm_list."),
            ("sigmoid1.", "sigmoid."),
            ("sigmoid1", "sigmoid"),
        )
        for k in list(state_dict.keys()):
            if not str(k).startswith(prefix):
                continue
            rel = str(k)[len(prefix) :]
            new_rel = None
            for old, new in legacy_map:
                if rel.startswith(old):
                    new_rel = new + rel[len(old) :]
                    break
                if rel == old:
                    new_rel = new
                    break
            if new_rel is None:
                continue
            new_key = prefix + new_rel
            if new_key not in state_dict:
                state_dict[new_key] = state_dict[k]
            del state_dict[k]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, snr, rate, model):
        B, C, H, W = x.size()
        device = x.device

        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        if model == "MambaVisionJSCC_w/o_SAandRA":
            x = self.head_list(x)
            return x

        elif model == "MambaVisionJSCC_w/_SA":
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
                    .expand(-1, x.size(1), -1)
                )
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            x = self.head_list(x)
            return x

        raise ValueError(f"Unsupported model variant (RA removed): {model}")

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
        for i_layer, layer in enumerate(self.layers):
            if isinstance(self.patch_size, (list, tuple)):
                patch_h, patch_w = int(self.patch_size[0]), int(self.patch_size[1])
            else:
                patch_h = patch_w = int(self.patch_size)
            base_h = H // patch_h
            base_w = W // patch_w
            if i_layer <= 1:
                layer.update_resolution(base_h, base_w)
            else:
                div = 2 ** (i_layer - 1)
                layer.update_resolution(base_h // div, base_w // div)


def create_mamba_encoder(**kwargs):
    return MambaJSCC_Encoder(**kwargs)
