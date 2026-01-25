import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choice

from net.encoder import create_mamba_encoder
from net.decoder import create_mamba_decoder
from net.channel import Channel
from loss.distortion import Distortion


class MambaVisionJSCC(nn.Module):
    """
    MambaVisionJSCC 网络：在 SwinJSCC 的 JSCC 框架下，
    使用 MambaVision 架构作为编码器 / 解码器骨干。
    """

    def __init__(self, args, config):
        super(MambaVisionJSCC, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs

        self.encoder = create_mamba_encoder(**encoder_kwargs)
        self.decoder = create_mamba_decoder(**decoder_kwargs)

        # Sanity checks: prevent silent channel-dim mismatch that can hurt PSNR and color fidelity.
        enc_last = int(encoder_kwargs["embed_dims"][-1])
        dec_first = int(decoder_kwargs["embed_dims"][0])
        if enc_last != dec_first:
            raise ValueError(
                "Encoder/Decoder channel mismatch: "
                f"encoder_last_dim={enc_last} != decoder_first_dim={dec_first}. "
                "Please keep the last encoder embed dim equal to the first decoder embed dim."
            )
        if encoder_kwargs.get("C", None) is None or decoder_kwargs.get("C", None) is None:
            raise ValueError(
                "Invalid config: encoder/decoder `C` is None. "
                "Rate-adaptive (RA) has been removed, so `--C` must be a single integer bottleneck dim."
            )

        if config.logger is not None:
            def _brief_kwargs(kwargs: dict) -> dict:
                keys = [
                    "img_size",
                    "patch_size",
                    "in_chans",
                    "embed_dims",
                    "depths",
                    "num_heads",
                    "window_size",
                    "mlp_ratio",
                    "patch_norm",
                    "drop_path_rate",
                    "layer_scale",
                    "layer_scale_conv",
                    "conv_token_mlp_ratio",
                    "post_upsample_attn_depth",
                    "refine_head",
                    "refine_channels",
                    "refine_scale",
                    "attn_gate_swin",
                    "attn_gate_mv",
                    "attn_gate",
                    "mamba_d_state",
                    "mamba_d_conv",
                    "mamba_expand",
                ]
                return {k: kwargs.get(k) for k in keys if k in kwargs}

            config.logger.info("MambaVisionJSCC backbone summary:")
            config.logger.info(f"Encoder: {_brief_kwargs(encoder_kwargs)}")
            config.logger.info(f"Decoder: {_brief_kwargs(decoder_kwargs)}")

        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        # NOTE: pass_channel can be toggled during training (warmup pretrain), so always consult config.
        self.squared_difference = torch.nn.MSELoss(reduction="none")
        self.H = self.W = 0

        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        # RA removed: still keep list for compatibility, but it should contain only one int.
        self.channel_number = [int(str(args.C).split(",")[0])]

        self.downsample = config.downsample
        self.model = args.model

    def distortion_loss_wrapper(self, x_gen, x_real):
        return self.distortion_loss.forward(
            x_gen, x_real, normalization=self.config.norm
        )

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    @staticmethod
    def _clamp_preserve_grad(x: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
        """
        Clamp in forward, but keep identity gradients (straight-through).

        This avoids "dead" gradients when x goes out of range, which can lead to
        dull/grey reconstructions for image tasks.
        """
        x_clamped = x.clamp(min_val, max_val)
        return x + (x_clamped - x).detach()

    def forward(self, input_image, given_SNR=None, given_rate=None):
        B, _, H, W = input_image.shape

        # Some DIV2K / Kodak images are not perfectly divisible by the overall downsample factor.
        # Pad to the nearest multiple to keep PatchMerging / reverse-merge shapes consistent,
        # then crop back to the original size after reconstruction.
        multiple = int(2 ** int(self.downsample))
        pad_h = (multiple - (H % multiple)) % multiple
        pad_w = (multiple - (W % multiple)) % multiple
        if pad_h > 0 or pad_w > 0:
            pad_mode = "reflect" if (H > 1 and W > 1) else "constant"
            input_pad = F.pad(input_image, (0, pad_w, 0, pad_h), mode=pad_mode)
        else:
            input_pad = input_image

        Hp, Wp = int(input_pad.shape[-2]), int(input_pad.shape[-1])
        if Hp != self.H or Wp != self.W:
            self.encoder.update_resolution(Hp, Wp)
            self.decoder.update_resolution(
                Hp // (2 ** self.downsample), Wp // (2 ** self.downsample)
            )
            self.H = Hp
            self.W = Wp

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if given_rate is None:
            channel_number = int(self.channel_number[0])
        else:
            channel_number = int(given_rate)
            fixed_c = int(self.channel_number[0])
            if channel_number != fixed_c:
                raise ValueError(
                    f"Rate-adaptive (RA) has been removed, so `given_rate` must equal the fixed bottleneck dim C. "
                    f"Got given_rate={channel_number}, but model C={fixed_c}."
                )

        if self.model not in ("MambaVisionJSCC_w/o_SAandRA", "MambaVisionJSCC_w/_SA"):
            raise ValueError(f"Unsupported model variant (RA removed): {self.model}")

        feature = self.encoder(input_pad, chan_param, channel_number, self.model)
        CBR = feature.numel() / 2 / input_image.numel()
        if bool(getattr(self.config, "pass_channel", True)):
            noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature

        recon_image = self.decoder(noisy_feature, chan_param, self.model)
        if pad_h > 0 or pad_w > 0:
            recon_image = recon_image[..., :H, :W]

        # Metrics use clamped images (valid pixel range).
        recon_metric = recon_image.clamp(0.0, 1.0)
        input_f32 = input_image.float()
        recon_metric_f32 = recon_metric.float()
        mse = self.squared_difference(input_f32 * 255.0, recon_metric_f32 * 255.0)

        # Loss is computed on a clamped image to match the PSNR/MSSSIM evaluation domain [0,1].
        # Use a "preserve-grad" clamp so gradients still flow for in-range pixels.
        recon_for_loss = self._clamp_preserve_grad(recon_image, 0.0, 1.0)

        loss_G = self.distortion_loss.forward(input_f32, recon_for_loss.float())
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()

    def forward_recon(self, input_image, given_SNR=None, given_rate=None):
        """
        Inference-only forward:
        returns reconstructed image and channel stats, without computing any loss.

        Useful for profiling FLOPs / latency and for tiled full-res evaluation helpers.
        """
        B, _, H, W = input_image.shape

        multiple = int(2 ** int(self.downsample))
        pad_h = (multiple - (H % multiple)) % multiple
        pad_w = (multiple - (W % multiple)) % multiple
        if pad_h > 0 or pad_w > 0:
            pad_mode = "reflect" if (H > 1 and W > 1) else "constant"
            input_pad = F.pad(input_image, (0, pad_w, 0, pad_h), mode=pad_mode)
        else:
            input_pad = input_image

        Hp, Wp = int(input_pad.shape[-2]), int(input_pad.shape[-1])
        if Hp != self.H or Wp != self.W:
            self.encoder.update_resolution(Hp, Wp)
            self.decoder.update_resolution(
                Hp // (2 ** self.downsample), Wp // (2 ** self.downsample)
            )
            self.H = Hp
            self.W = Wp

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if given_rate is None:
            channel_number = int(self.channel_number[0])
        else:
            channel_number = int(given_rate)
            fixed_c = int(self.channel_number[0])
            if channel_number != fixed_c:
                raise ValueError(
                    f"Rate-adaptive (RA) has been removed, so `given_rate` must equal the fixed bottleneck dim C. "
                    f"Got given_rate={channel_number}, but model C={fixed_c}."
                )

        if self.model not in ("MambaVisionJSCC_w/o_SAandRA", "MambaVisionJSCC_w/_SA"):
            raise ValueError(f"Unsupported model variant (RA removed): {self.model}")

        feature = self.encoder(input_pad, chan_param, channel_number, self.model)
        CBR = feature.numel() / 2 / input_image.numel()
        if bool(getattr(self.config, "pass_channel", True)):
            noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature

        recon_image = self.decoder(noisy_feature, chan_param, self.model)
        if pad_h > 0 or pad_w > 0:
            recon_image = recon_image[..., :H, :W]
        return recon_image, CBR, chan_param
