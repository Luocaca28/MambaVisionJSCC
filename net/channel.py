import torch.nn as nn
import numpy as np
import torch


class Channel(nn.Module):
    """
    Channel model: supports error free, Rayleigh and AWGN channels.
    API is kept compatible with the original SwinJSCC implementation.
    """

    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = args.channel_type
        self.device = config.device
        self.h = (
            torch.sqrt(torch.randn(1) ** 2 + torch.randn(1) ** 2) / 1.414
        )
        if config.logger:
            config.logger.info(
                "【Channel】 Built {} channel, SNR {} dB.".format(
                    args.channel_type, args.multiple_snr
                )
            )

    def gaussian_noise_layer(self, input_layer, std, name=None):
        device = input_layer.get_device()
        noise_real = torch.normal(
            mean=0.0, std=std, size=np.shape(input_layer), device=device
        )
        noise_imag = torch.normal(
            mean=0.0, std=std, size=np.shape(input_layer), device=device
        )
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer))
        noise = noise_real + 1j * noise_imag
        h = (
            torch.sqrt(
                torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2
                + torch.normal(
                    mean=0.0, std=1, size=np.shape(input_layer)
                )
                ** 2
            )
            / np.sqrt(2)
        )
        if self.config.CUDA:
            noise = noise.to(input_layer.get_device())
            h = h.to(input_layer.get_device())
        return input_layer * h + noise

    def complex_normalize(self, x, power):
        """
        Normalize real-valued tensor `x` (which represents concatenated real/imag parts)
        to satisfy an average complex-symbol power constraint.

        We compute power in float32 for numerical stability under AMP, and per-sample
        (per batch element) to avoid brightness/contrast drift across images.
        """
        x_f32 = x.float()
        # Reduce over non-batch dims, keep dims for broadcasting.
        reduce_dims = tuple(range(1, x_f32.dim())) if x_f32.dim() > 1 else ()
        pwr = x_f32.pow(2).mean(dim=reduce_dims, keepdim=True) * 2.0
        # Avoid divide-by-zero for degenerate inputs.
        pwr = pwr.clamp_min(1e-12)

        scale = (float(np.sqrt(power)) / torch.sqrt(pwr)).to(dtype=x.dtype)
        out = x * scale
        return out, pwr

    def forward(self, input, chan_param, avg_pwr=None):
        use_avg = avg_pwr is not None and avg_pwr is not False
        if use_avg:
            power = 1
            # avg_pwr may be a scalar or per-sample tensor; compute scaling in float32 for stability.
            avg_pwr_f32 = avg_pwr.float() if torch.is_tensor(avg_pwr) else torch.tensor(float(avg_pwr), device=input.device)
            denom = torch.sqrt(avg_pwr_f32 * 2.0).clamp_min(1e-12)
            channel_tx = (float(np.sqrt(power)) * input) / denom.to(dtype=input.dtype)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)
        input_shape = channel_tx.shape
        channel_tx_dtype = channel_tx.dtype
        device_type = "cuda" if channel_tx.is_cuda else "cpu"
        # Force the complex channel math to float32 to avoid ComplexHalf warnings under AMP.
        with torch.amp.autocast(device_type=device_type, enabled=False):
            channel_in = channel_tx.float().reshape(-1)
            L = channel_in.shape[0]
            channel_in = channel_in[: L // 2] + channel_in[L // 2 :] * 1j
            channel_output = self.complex_forward(channel_in, chan_param)
            channel_output = torch.cat(
                [torch.real(channel_output), torch.imag(channel_output)]
            )
            channel_output = channel_output.reshape(input_shape)
        channel_output = channel_output.to(dtype=channel_tx_dtype)
        if self.chan_type == 1 or self.chan_type == "awgn":
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise
            if use_avg:
                avg_pwr_f32 = avg_pwr.float() if torch.is_tensor(avg_pwr) else torch.tensor(float(avg_pwr), device=channel_tx.device)
                scale_back = torch.sqrt(avg_pwr_f32 * 2.0)
                return channel_tx * scale_back.to(dtype=channel_tx.dtype)
            else:
                return channel_tx * torch.sqrt(pwr).to(dtype=channel_tx.dtype)
        elif self.chan_type == 2 or self.chan_type == "rayleigh":
            if use_avg:
                avg_pwr_f32 = avg_pwr.float() if torch.is_tensor(avg_pwr) else torch.tensor(float(avg_pwr), device=channel_output.device)
                scale_back = torch.sqrt(avg_pwr_f32 * 2.0)
                return channel_output * scale_back.to(dtype=channel_output.dtype)
            else:
                return channel_output * torch.sqrt(pwr).to(dtype=channel_output.dtype)

    def complex_forward(self, channel_in, chan_param):
        if self.chan_type == 0 or self.chan_type == "none":
            return channel_in

        elif self.chan_type == 1 or self.chan_type == "awgn":
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.gaussian_noise_layer(
                channel_tx, std=sigma, name="awgn_chan_noise"
            )
            return chan_output

        elif self.chan_type == 2 or self.chan_type == "rayleigh":
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(
                channel_tx, std=sigma, name="rayleigh_chan_noise"
            )
            return chan_output

    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx
