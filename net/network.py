from random import choice

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.distortion import Distortion
from net.channel import Channel
from net.decoder import create_mamba_decoder
from net.encoder import create_mamba_encoder
from net.snr_film import SNRFiLM


class MambaVisionJSCC(nn.Module):
    """End-to-end image JSCC model with a MambaVision backbone."""

    def __init__(self, args, config):
        super().__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs

        self.encoder = create_mamba_encoder(**encoder_kwargs)
        self.decoder = create_mamba_decoder(**decoder_kwargs)

        enc_last = int(encoder_kwargs["embed_dims"][-1])
        dec_first = int(decoder_kwargs["embed_dims"][0])
        if enc_last != dec_first:
            raise ValueError(
                "Encoder/Decoder channel mismatch: "
                f"encoder_last_dim={enc_last} != decoder_first_dim={dec_first}."
            )
        if encoder_kwargs.get("C") is None or decoder_kwargs.get("C") is None:
            raise ValueError("Encoder/decoder bottleneck dimension C must be set.")

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
                    "refine_head",
                    "refine_channels",
                    "refine_scale",
                    "attn_gate_swin",
                    "attn_gate_mv",
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
        self.squared_difference = nn.MSELoss(reduction="none")
        self.H = 0
        self.W = 0
        self.multiple_snr = [int(snr) for snr in str(args.multiple_snr).split(",")]
        self.channel_number = int(str(args.C).split(",")[0])
        self.downsample = config.downsample
        self.snr_film_position = str(getattr(config, "snr_film_position", "none"))
        self.use_snr_film = self.snr_film_position != "none"

        snr_min = float(min(self.multiple_snr))
        snr_max = float(max(self.multiple_snr))
        snr_hidden = int(getattr(config, "snr_film_hidden", 64))
        snr_film_scale = float(getattr(config, "snr_film_scale", 0.1))

        if self.use_snr_film:
            self.enc_snr_film = SNRFiLM(
                dim=self.channel_number,
                hidden_dim=snr_hidden,
                snr_min=snr_min,
                snr_max=snr_max,
                film_scale=snr_film_scale,
                identity_init=True,
            )
            self.dec_snr_film = SNRFiLM(
                dim=self.channel_number,
                hidden_dim=snr_hidden,
                snr_min=snr_min,
                snr_max=snr_max,
                film_scale=snr_film_scale,
                identity_init=True,
            )
        else:
            self.enc_snr_film = None
            self.dec_snr_film = None

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        return self.channel(feature, chan_param, avg_pwr)

    @staticmethod
    def _clamp_preserve_grad(
        x: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0
    ) -> torch.Tensor:
        x_clamped = x.clamp(min_val, max_val)
        return x + (x_clamped - x).detach()

    def forward(self, input_image, given_SNR=None, given_rate=None):
        _, _, H, W = input_image.shape

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

        chan_param = choice(self.multiple_snr) if given_SNR is None else int(given_SNR)
        if given_rate is not None and int(given_rate) != self.channel_number:
            raise ValueError(
                f"Fixed-rate model expects C={self.channel_number}, got {int(given_rate)}."
            )

        feature = self.encoder(input_pad)

        if self.snr_film_position in ["enc", "both"]:
            feature = self.enc_snr_film(feature, chan_param)

        cbr = feature.numel() / 2 / input_image.numel()
        noisy_feature = (
            self.feature_pass_channel(feature, chan_param)
            if bool(getattr(self.config, "pass_channel", True))
            else feature
        )

        if self.snr_film_position in ["dec", "both"]:
            noisy_feature = self.dec_snr_film(noisy_feature, chan_param)

        recon_image = self.decoder(noisy_feature)
        if pad_h > 0 or pad_w > 0:
            recon_image = recon_image[..., :H, :W]

        recon_metric = recon_image.clamp(0.0, 1.0)
        input_f32 = input_image.float()
        recon_metric_f32 = recon_metric.float()
        mse = self.squared_difference(input_f32 * 255.0, recon_metric_f32 * 255.0)

        recon_for_loss = self._clamp_preserve_grad(recon_image, 0.0, 1.0)
        loss_G = self.distortion_loss(input_f32, recon_for_loss.float())
        return recon_image, cbr, chan_param, mse.mean(), loss_G.mean()
