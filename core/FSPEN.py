"""
# File : sequence_modules.py
# Author : wukeyi
# version : python3.9
"""

from typing import Dict, List, Tuple
from einops import rearrange

import torch
from pydantic import BaseModel, field_validator
from torch import Tensor, nn

from JointNSHModel import expand_HT
from models.conv_stft import STFT
from utils.check_flops import check_flops


def get_sub_bands(band_parameters: dict):
    group_bands = list()
    group_band_width = list()
    for key, value in band_parameters.items():
        num_band = (
            value["group_width"] - value["conv"]["kernel_size"] + 2 * value["conv"]["padding"]
        ) // value["conv"]["stride"] + 1
        sub_band_width = value["group_width"] // num_band
        group_bands.append(num_band)
        group_band_width.append(sub_band_width)

    return tuple(group_bands), tuple(group_band_width)


class TrainConfig(BaseModel):
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    train_frames: int = 62
    train_points: int = (train_frames - 1) * hop_length

    full_band_encoder: Dict[str, dict] = {
        "encoder1": {
            "in_channels": 2,
            "out_channels": 4,
            "kernel_size": 6,
            "stride": 2,
            "padding": 2,
        },
        "encoder2": {
            "in_channels": 4,
            "out_channels": 16,
            "kernel_size": 8,
            "stride": 2,
            "padding": 3,
        },
        "encoder3": {
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": 6,
            "stride": 2,
            "padding": 2,
        },
    }
    full_band_decoder: Dict[str, dict] = {
        "decoder1": {
            "in_channels": 64,
            "out_channels": 16,
            "kernel_size": 6,
            "stride": 2,
            "padding": 2,
        },
        "decoder2": {
            "in_channels": 32,
            "out_channels": 4,
            "kernel_size": 8,
            "stride": 2,
            "padding": 3,
        },
        "decoder3": {
            "in_channels": 8,
            "out_channels": 2,
            "kernel_size": 6,
            "stride": 2,
            "padding": 2,
        },
    }

    sub_band_encoder: Dict[str, dict] = {
        "encoder1": {
            "group_width": 16,
            "conv": {
                "start_frequency": 0,
                "end_frequency": 16,
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 4,
                "stride": 2,
                "padding": 1,
            },
        },
        "encoder2": {
            "group_width": 18,
            "conv": {
                "start_frequency": 16,
                "end_frequency": 34,
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 7,
                "stride": 3,
                "padding": 2,
            },
        },
        "encoder3": {
            "group_width": 36,
            "conv": {
                "start_frequency": 34,
                "end_frequency": 70,
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 11,
                "stride": 5,
                "padding": 2,
            },
        },
        "encoder4": {
            "group_width": 66,
            "conv": {
                "start_frequency": 70,
                "end_frequency": 136,
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 20,
                "stride": 10,
                "padding": 4,
            },
        },
        "encoder5": {
            "group_width": 121,
            "conv": {
                "start_frequency": 136,
                "end_frequency": 257,
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 30,
                "stride": 20,
                "padding": 5,
            },
        },
    }
    merge_split: dict = {"channels": 64, "bands": 32, "compress_rate": 2}
    bands_num_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[0]
    band_width_in_groups: Tuple[int] = get_sub_bands(sub_band_encoder)[1]

    sub_band_decoder: Dict[str, dict] = {
        f"decoder{idx}": {"in_features": 64, "out_features": width}
        for idx, width in enumerate(band_width_in_groups)
    }

    dual_path_extension: dict = {
        "num_modules": 3,
        "parameters": {
            "input_size": 16,
            "intra_hidden_size": 16,
            "inter_hidden_size": 16,
            "groups": 8,
            "rnn_type": "GRU",
        },
    }

    @field_validator("sub_band_decoder")
    def sub_band_decoder_validate(cls, decoders):
        for decoder in decoders:
            if decoder["out_feature"] < 2:
                raise ValueError(f"values should > 2, but got {decoder['out_feature']}")


class FullBandEncoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.norm = nn.BatchNorm1d(num_features=out_channels)

        self.activate = nn.ELU()

    def forward(self, complex_spectrum: Tensor):
        """
        :param complex_spectrum: (batch * frames, channels, frequency)
        :return:
        """
        complex_spectrum = self.conv(complex_spectrum)
        complex_spectrum = self.norm(complex_spectrum)
        complex_spectrum = self.activate(complex_spectrum)

        return complex_spectrum


class FullBandDecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.convT = nn.ConvTranspose1d(
            in_channels // 2, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

        self.norm = nn.BatchNorm1d(num_features=out_channels)
        self.activate = nn.ELU()

    def forward(self, encode_complex_spectrum: Tensor, decode_complex_spectrum):
        """
        :param decode_complex_spectrum: (batch * frames, channels1, frequency)
        :param encode_complex_spectrum: (batch * frames, channels2, frequency)
        :return:
        """
        complex_spectrum = torch.cat([encode_complex_spectrum, decode_complex_spectrum], dim=1)
        complex_spectrum = self.conv(complex_spectrum)
        complex_spectrum = self.convT(complex_spectrum)
        complex_spectrum = self.norm(complex_spectrum)
        complex_spectrum = self.activate(complex_spectrum)

        return complex_spectrum


class SubBandEncoderBlock(nn.Module):
    def __init__(
        self,
        start_frequency: int,
        end_frequency: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activate = nn.ReLU()

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch*frames, channels, frequency)
        :return:
        """
        sub_spectrum = amplitude_spectrum[:, :, self.start_frequency : self.end_frequency]

        sub_spectrum = self.conv(sub_spectrum)  # (batch*frames, out_channels, sub_bands)
        sub_spectrum = self.activate(sub_spectrum)

        return sub_spectrum


class SubBandDecoderBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, start_idx: int, end_idx: int):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activate = nn.ReLU()

    def forward(self, encode_amplitude_spectrum: Tensor, decode_amplitude_spectrum: Tensor):
        """

        :param encode_amplitude_spectrum: (batch * frames, channels, sub_bands)
        :param decode_amplitude_spectrum: (batch * frames, channels, sub_bands)
        :return:
        """
        encode_amplitude_spectrum = encode_amplitude_spectrum[:, :, self.start_idx : self.end_idx]
        spectrum = torch.cat(
            [encode_amplitude_spectrum, decode_amplitude_spectrum], dim=1
        )  # channels cat
        spectrum = torch.transpose(spectrum, dim0=1, dim1=2).contiguous()  # (*, bands, channels)

        spectrum = self.fc(spectrum)  # (*, bands, band-width)
        spectrum = self.activate(spectrum)
        first_dim, bands, band_width = spectrum.shape
        spectrum = torch.reshape(spectrum, shape=(first_dim, bands * band_width))

        return spectrum


class GroupRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int,
        rnn_type: str,
        num_layers: int = 1,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        assert (
            input_size % groups == 0
        ), f"input_size % groups must be equal to 0, but got {input_size} % {groups} = {input_size % groups}"

        self.groups = groups
        self.rnn_list = nn.ModuleList()
        for _ in range(groups):
            self.rnn_list.append(
                getattr(nn, rnn_type)(
                    input_size=input_size // groups,
                    hidden_size=hidden_size // groups,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    batch_first=batch_first,
                )
            )

    def forward(self, inputs: Tensor, hidden_state: List[Tensor]):
        """
        :param hidden_state: List[state1, state2, ...], len(hidden_state) = groups
        state shape = (num_layers*bidirectional, batch*[], hidden_size) if rnn_type is GRU or RNN, otherwise,
        state = (h0, c0), h0/c0 shape = (num_layers*bidirectional, batch*[], hidden_size).
        :param inputs: (batch, steps, input_size)
        :return:
        """
        outputs = []
        out_states = []
        batch, steps, _ = inputs.shape

        inputs = torch.reshape(
            inputs, shape=(batch, steps, self.groups, -1)
        )  # (batch, steps, groups, width)
        for idx, rnn in enumerate(self.rnn_list):
            out, state = rnn(inputs[:, :, idx, :], hidden_state[idx])
            outputs.append(out)  # (batch, steps, hidden_size)
            out_states.append(state)  # (num_layers*bidirectional, batch*[], hidden_size)

        outputs = torch.cat(outputs, dim=2)  # (batch, steps, hidden_size * groups)

        return outputs, out_states


class DualPathExtensionRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        intra_hidden_size: int,
        inter_hidden_size: int,
        groups: int,
        rnn_type: str,
    ):
        super().__init__()
        assert rnn_type in [
            "RNN",
            "GRU",
            "LSTM",
        ], f"rnn_type should be RNN/GRU/LSTM, but got {rnn_type}!"

        self.intra_chunk_rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=intra_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.intra_chunk_fc = nn.Linear(in_features=intra_hidden_size * 2, out_features=input_size)
        self.intra_chunk_norm = nn.LayerNorm(normalized_shape=input_size, elementwise_affine=True)

        self.inter_chunk_rnn = GroupRNN(
            input_size=input_size, hidden_size=inter_hidden_size, groups=groups, rnn_type=rnn_type
        )
        self.inter_chunk_fc = nn.Linear(in_features=inter_hidden_size, out_features=input_size)

    def forward(self, inputs: Tensor, hidden_state: List[Tensor]):
        """
        :param hidden_state: List[state1, state2, ...], len(hidden_state) = groups
        state shape = (num_layers*bidirectional, batch*[], hidden_size) if rnn_type is GRU or RNN, otherwise,
        state = (h0, c0), h0/c0 shape = (num_layers*bidirectional, batch*[], hidden_size).
        :param inputs: (B, F, T, N)
        :return:
        """
        B, F, T, N = inputs.shape
        intra_out = torch.transpose(inputs, dim0=1, dim1=2).contiguous()  # (B, T, F, N)
        intra_out = torch.reshape(intra_out, shape=(B * T, F, N))
        intra_out, _ = self.intra_chunk_rnn(intra_out)
        intra_out = self.intra_chunk_fc(intra_out)  # (B, T, F, N)
        intra_out = torch.reshape(intra_out, shape=(B, T, F, N))
        intra_out = torch.transpose(intra_out, dim0=1, dim1=2).contiguous()  # (B, F, T, N)
        intra_out = self.intra_chunk_norm(intra_out)  # (B, F, T, N)

        intra_out = inputs + intra_out  # residual add

        inter_out = torch.reshape(intra_out, shape=(B * F, T, N))  # (B*F, T, N)
        inter_out, hidden_state = self.inter_chunk_rnn(inter_out, hidden_state)
        inter_out = torch.reshape(inter_out, shape=(B, F, T, -1))  # (B, F, T, groups * N)
        inter_out = self.inter_chunk_fc(inter_out)  # (B, F, T, N)

        inter_out = inter_out + intra_out  # residual add

        return inter_out, hidden_state


class FullBandEncoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        last_channels = 0
        self.full_band_encoder = nn.ModuleList()
        for encoder_name, conv_parameter in configs.full_band_encoder.items():
            self.full_band_encoder.append(FullBandEncoderBlock(**conv_parameter))
            last_channels = conv_parameter["out_channels"]

        self.global_features = nn.Conv1d(
            in_channels=last_channels, out_channels=last_channels, kernel_size=1, stride=1
        )

    def forward(self, complex_spectrum: Tensor):
        """
        :param complex_spectrum: (batch*frame, channels, frequency)
        :return:
        """
        full_band_encodes = []
        for encoder in self.full_band_encoder:
            complex_spectrum = encoder(complex_spectrum)
            full_band_encodes.append(complex_spectrum)

        global_feature = self.global_features(complex_spectrum)

        return full_band_encodes[::-1], global_feature


class SubBandEncoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        self.sub_band_encoders = nn.ModuleList()
        for encoder_name, conv_parameters in configs.sub_band_encoder.items():
            self.sub_band_encoders.append(SubBandEncoderBlock(**conv_parameters["conv"]))

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch * frames, channels, frequency)
        :return:
        """
        sub_band_encodes = list()
        for encoder in self.sub_band_encoders:
            encode_out = encoder(amplitude_spectrum)
            sub_band_encodes.append(encode_out)

        local_feature = torch.cat(sub_band_encodes, dim=2)  # feature cat

        return sub_band_encodes, local_feature


class FullBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        self.full_band_decoders = nn.ModuleList()
        for decoder_name, parameters in configs.full_band_decoder.items():
            self.full_band_decoders.append(FullBandDecoderBlock(**parameters))

    def forward(self, feature: Tensor, encode_outs: list):
        for decoder, encode_out in zip(self.full_band_decoders, encode_outs):
            feature = decoder(feature, encode_out)

        return feature


class SubBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        start_idx = 0
        self.sub_band_decoders = nn.ModuleList()
        for (decoder_name, parameters), bands in zip(
            configs.sub_band_decoder.items(), configs.bands_num_in_groups
        ):
            end_idx = start_idx + bands
            self.sub_band_decoders.append(
                SubBandDecoderBlock(start_idx=start_idx, end_idx=end_idx, **parameters)
            )

    def forward(self, feature: Tensor, sub_encodes: list):
        """
        :param feature: (batch*frames, channels, bands)
        :param sub_encodes: [sub_encode_0, sub_encode_1, ...], each element is (batch*frames, channels, sub_bands)
        :return: (batch*frames, full-frequency)
        """
        sub_decoder_outs = []
        for decoder, sub_encode in zip(self.sub_band_decoders, sub_encodes):
            sub_decoder_out = decoder(feature, sub_encode)
            sub_decoder_outs.append(sub_decoder_out)

        sub_decoder_outs = torch.cat(tensors=sub_decoder_outs, dim=1)  # feature cat

        return sub_decoder_outs


class FullSubPathExtension(nn.Module):
    def __init__(self, configs: TrainConfig = TrainConfig()):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        self.sub_band_encoder = SubBandEncoder(configs)

        merge_split = configs.merge_split
        merge_channels = merge_split["channels"]
        merge_bands = merge_split["bands"]
        compress_rate = merge_split["compress_rate"]

        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=merge_channels, out_features=merge_channels // compress_rate),
            nn.ELU(),
            nn.Conv1d(
                in_channels=merge_bands,
                out_channels=merge_bands // compress_rate,
                kernel_size=1,
                stride=1,
            ),
        )

        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(
                DualPathExtensionRNN(**configs.dual_path_extension["parameters"])
            )

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=merge_bands // compress_rate,
                out_channels=merge_bands,
                kernel_size=1,
                stride=1,
            ),
            nn.Linear(in_features=merge_channels // compress_rate, out_features=merge_channels),
            nn.ELU(),
        )

        self.full_band_decoder = FullBandDecoder(configs)
        self.sub_band_decoder = SubBandDecoder(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.configs = configs

        self.stft = STFT(configs.n_fft, configs.hop_length, configs.n_fft)

    def init_hidden_state(self, x):
        nB = x.size(0)
        num_bands = sum(self.configs.bands_num_in_groups)
        num_modules = self.configs.dual_path_extension["num_modules"]
        inter_hidden_size = self.configs.dual_path_extension["parameters"]["inter_hidden_size"]
        groups = self.configs.dual_path_extension["parameters"]["groups"]
        in_hidden_state = [
            [
                torch.zeros(1, nB * num_bands, inter_hidden_size // groups).to(x.device)
                for _ in range(groups)
            ]
            for _ in range(num_modules)
        ]
        return in_hidden_state

    # def forward(
    #     self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list
    # ):
    #     """
    #     :param in_amplitude_spectrum: (batch, frames, 1, frequency)
    #     :param hidden_state:
    #     :param in_complex_spectrum: (batch, frames, channels, frequency)
    #     :return:
    #     """
    def forward(self, inp: Tensor, hidden_state: list = []):
        xk = self.stft.transform(inp)
        mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # b,1,t,f
        in_complex_spectrum = xk.permute(0, 2, 1, 3)  # b,t,c,f
        in_amplitude_spectrum = mag.permute(0, 2, 1, 3)

        if hidden_state == []:
            hidden_state = self.init_hidden_state(inp)

        batch, frames, channels, frequency = in_complex_spectrum.shape
        complex_spectrum = torch.reshape(
            in_complex_spectrum, shape=(batch * frames, channels, frequency)
        )
        amplitude_spectrum = torch.reshape(
            in_amplitude_spectrum, shape=(batch * frames, 1, frequency)
        )

        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)

        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        merge_feature = self.feature_merge_layer(merge_feature)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        out_hidden_state = list()
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)

        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))

        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))

        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        full_band_mask = self.mask_padding(full_band_mask)
        sub_band_mask = self.mask_padding(sub_band_mask)

        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        spec = full_band_out + sub_band_out
        spec = spec.permute(0, 2, 1, 3)
        out = self.stft.inverse(spec)
        # return full_band_out + sub_band_out, out_hidden_state
        return out, out_hidden_state


class FullSubPathExtensionFIG6(nn.Module):
    def __init__(self, configs: TrainConfig = TrainConfig()):
        super().__init__()

        configs.full_band_encoder["encoder1"]["in_channels"] = 3
        # configs.sub_band_encoder["encoder1"]["conv"]["in_channels"] = 2

        self.full_band_encoder = FullBandEncoder(configs)
        self.sub_band_encoder = SubBandEncoder(configs)

        merge_split = configs.merge_split
        merge_channels = merge_split["channels"]
        merge_bands = merge_split["bands"]
        compress_rate = merge_split["compress_rate"]

        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=merge_channels, out_features=merge_channels // compress_rate),
            nn.ELU(),
            nn.Conv1d(
                in_channels=merge_bands,
                out_channels=merge_bands // compress_rate,
                kernel_size=1,
                stride=1,
            ),
        )

        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(
                DualPathExtensionRNN(**configs.dual_path_extension["parameters"])
            )

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=merge_bands // compress_rate,
                out_channels=merge_bands,
                kernel_size=1,
                stride=1,
            ),
            nn.Linear(in_features=merge_channels // compress_rate, out_features=merge_channels),
            nn.ELU(),
        )

        self.full_band_decoder = FullBandDecoder(configs)
        self.sub_band_decoder = SubBandDecoder(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)
        self.configs = configs

        self.stft = STFT(configs.n_fft, configs.hop_length, configs.n_fft)

        self.reso = 16000 / configs.n_fft

    def init_hidden_state(self, x):
        nB = x.size(0)
        num_bands = sum(self.configs.bands_num_in_groups)
        num_modules = self.configs.dual_path_extension["num_modules"]
        inter_hidden_size = self.configs.dual_path_extension["parameters"]["inter_hidden_size"]
        groups = self.configs.dual_path_extension["parameters"]["groups"]
        in_hidden_state = [
            [
                torch.zeros(1, nB * num_bands, inter_hidden_size // groups).to(x.device)
                for _ in range(groups)
            ]
            for _ in range(num_modules)
        ]
        return in_hidden_state

    # def forward(
    #     self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list
    # ):
    #     """
    #     :param in_amplitude_spectrum: (batch, frames, 1, frequency)
    #     :param hidden_state:
    #     :param in_complex_spectrum: (batch, frames, channels, frequency)
    #     :return:
    #     """
    def forward(self, inp: Tensor, HL, hidden_state: list = []):
        xk = self.stft.transform(inp)
        hl = expand_HT(HL, xk.shape[-2], self.reso)  # B,C(1),T,F
        mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # b,1,t,f

        in_complex_spectrum = xk.permute(0, 2, 1, 3)  # b,t,c,f
        in_amplitude_spectrum = mag.permute(0, 2, 1, 3)

        if hidden_state == []:
            hidden_state = self.init_hidden_state(inp)

        batch, frames, channels, frequency = in_complex_spectrum.shape
        complex_spectrum = torch.reshape(
            in_complex_spectrum, shape=(batch * frames, channels, frequency)
        )
        amplitude_spectrum = torch.reshape(
            in_amplitude_spectrum, shape=(batch * frames, 1, frequency)
        )

        hl = rearrange(hl, "b c t f->(b t) c f")
        complex_spectrum = torch.concat([complex_spectrum, hl], dim=1)
        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)

        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
        merge_feature = self.feature_merge_layer(merge_feature)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        out_hidden_state = list()
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)

        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))

        split_feature = self.feature_split_layer(merge_feature)
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))

        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        full_band_mask = self.mask_padding(full_band_mask)
        sub_band_mask = self.mask_padding(sub_band_mask)

        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        spec = full_band_out + sub_band_out
        spec = spec.permute(0, 2, 1, 3)
        out = self.stft.inverse(spec)
        # return full_band_out + sub_band_out, out_hidden_state
        # return out, out_hidden_state
        return out


if __name__ == "__main__":
    # inp = torch.randn(1, 16000)
    # net = FullSubPathExtension()
    # out, _ = net(inp)
    # print(out.shape)

    # check_flops(net, inp)

    inp = torch.randn(1, 16000)
    hl = torch.randn(1, 6)
    net = FullSubPathExtensionFIG6()
    out, _ = net(inp, hl)
    print(out.shape)

    check_flops(net, inp, hl)
