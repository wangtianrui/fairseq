# coding: utf-8
# Author：WangTianRui
# Date ：2021-05-21 10:19
from email.policy import default
import os ,sys

from importlib_metadata import metadata
import torch
import math
import numpy as np
from .utils import ConvSTFT, ConviSTFT
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model


@dataclass
class DCRNConfig(FairseqDataclass):
    rnn_hidden: int = field(default=128, metadata={"help": "lstm hidden"})
    win_len: int = field(default=512, metadata={"help": "stft win len"})
    hop_len: int = field(default=128, metadata={"help": "stft hop len"})
    fft_len: int = field(default=512, metadata={"help": "stft nfft len"})
    win_type: str = field(default="hanning", metadata={"help": "stft win type"})
    kernel_size: int = field(default=5, metadata={"help": "conv kernel size of frequency"})
    kernel_num: str = field(default="16, 32, 64, 128, 128, 128", metadata={"help": "ker num"})


def complex_cat(x1, x2):
    x1_real, x1_imag = torch.chunk(x1, 2, dim=1)
    x2_real, x2_imag = torch.chunk(x2, 2, dim=1)
    return torch.cat(
        [torch.cat([x1_real, x2_real], dim=1), torch.cat([x1_imag, x2_imag], dim=1)], dim=1
    )


class CausalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(CausalConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.left_pad = kernel_size[1] - 1
        # padding = (kernel_size[0] // 2, 0)
        padding = (kernel_size[0] // 2, self.left_pad)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding)

    def forward(self, x):
        """
        :param x: B,C,F,T
        :return:
        """
        B, C, F, T = x.size()
        # x = F.pad(x, [self.left_pad, 0])
        return self.conv(x)[..., :T]


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding):
        super(CausalTransConvBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                                             stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        """
        因果反卷积
        :param x: B,C,F,T
        :return:
        """
        T = x.size(-1)
        conv_out = self.trans_conv(x)[..., :T]
        return conv_out


@register_model("dcrn", dataclass=DCRNConfig)
class DCRN(BaseFairseqModel):
    def __init__(self, cfg):
        super(DCRN, self).__init__()
        kernel_num = tuple([int(item) for item in  cfg.kernel_num.split(",")])
        self.rnn_hidden = cfg.rnn_hidden
        self.win_len = cfg.win_len
        self.hop_len = cfg.hop_len
        self.fft_len = cfg.fft_len
        self.win_type = cfg.win_type
        self.kernel_size = cfg.kernel_size
        self.kernel_num = (2,) + kernel_num

        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, self.win_type, 'complex')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, self.win_type, 'complex')

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    CausalConv(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))
        # print(hidden_dim)

        self.enhance = nn.LSTM(
            input_size=hidden_dim * self.kernel_num[-1],
            hidden_size=self.rnn_hidden,
            num_layers=1,
            dropout=0.0,
            batch_first=False
        )
        self.transform = nn.Linear(self.rnn_hidden, hidden_dim * self.kernel_num[-1])
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        CausalTransConvBlock(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        CausalTransConvBlock(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        )
                    )
                )
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg)

    def forward(self, noisy):
        stft = self.stft(noisy)
        real = stft[:, :self.fft_len // 2 + 1]
        imag = stft[:, self.fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_phase = torch.atan(imag / (real + 1e-8))
        phase_adjust = (real < 0).to(torch.int) * torch.sign(imag) * math.pi
        spec_phase = spec_phase + phase_adjust
        spec_complex = torch.stack([real, imag], dim=1)[:, :, 1:]  # B,2,256

        out = spec_complex
        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            encoder_out.append(out)

        B, C, D, T = out.size()
        out = out.permute(3, 0, 1, 2)
        out = torch.reshape(out, [T, B, C * D])
        out, _ = self.enhance(out)
        out = self.transform(out)
        out = torch.reshape(out, [T, B, C, D])
        out = out.permute(1, 2, 3, 0)

        for idx in range(len(self.decoder)):
            out = torch.cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0], value=1e-8)
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0], value=1e-8)
        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan(
            imag_phase / (real_phase + 1e-8)
        )
        phase_adjust = (real_phase < 0).to(torch.int) * torch.sign(imag_phase) * math.pi
        mask_phase = mask_phase + phase_adjust
        mask_mags = torch.tanh(mask_mags)
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        real = est_mags * torch.cos(est_phase)
        imag = est_mags * torch.sin(est_phase)
        out_spec = torch.cat([real, imag], 1)
        return self.istft(out_spec).squeeze(1)
