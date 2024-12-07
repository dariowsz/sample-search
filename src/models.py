from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from msclap import CLAP


def reshape_wav2img(self, x):
    B, C, T, F = x.shape
    target_T = int(self.spec_size * self.freq_ratio)
    target_F = self.spec_size // self.freq_ratio
    assert (
        T <= target_T and F <= target_F
    ), "the wav size should less than or equal to the swin input size"
    # to avoid bicubic zero error
    if T < target_T:
        x = nn.functional.interpolate(
            x, (target_T, x.shape[3]), mode="bilinear", align_corners=True
        )
    if F < target_F:
        x = nn.functional.interpolate(
            x, (x.shape[2], target_F), mode="bilinear", align_corners=True
        )
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.reshape(
        x.shape[0],
        x.shape[1],
        x.shape[2],
        self.freq_ratio,
        x.shape[3] // self.freq_ratio,
    )
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
    return x


class EncoderModel(ABC):
    @abstractmethod
    def __init__(self, weights_path: str):
        """Initialize the encoder model"""
        pass

    @abstractmethod
    def get_audio_embeddings(self, audio_files: list[str]) -> torch.Tensor:
        """Get audio embeddings"""
        pass

    @abstractmethod
    def get_text_embeddings(self, text: list[str]) -> torch.Tensor:
        """Get text embeddings"""
        pass


class MSClap(EncoderModel):
    def __init__(
        self, weights_path: str, device: Literal["cuda", "mps", "cpu"] = "cpu"
    ):
        self.model = CLAP(weights_path)
        self.device = device

        if self.device == "cuda" and torch.cuda.is_available():
            self.model.use_cuda = True
            self.model.clap.to(torch.device(self.device))
        elif self.device == "mps" and torch.backends.mps.is_available():
            # Temporary patch to use Bilinear interpolation instead of Bicubic because MPS doesn't support Bicubic
            # This creates a small performance hit (distribution shift) but works well in practice and is much faster
            print(
                "[WARNING]: CLAP uses Bicubic interpolation which is not supported on MPS. I'm forcing the model to use Bilinear interpolation instead for MPS compatibility and faster inference but performance will be slightly degraded."
            )
            print(
                "If you are running on a Silicon Mac and want the best performance, change the device to 'cpu', but be aware that model will be considerably slower."
            )
            self.model.clap.audio_encoder.base.htsat.reshape_wav2img = (
                lambda x: reshape_wav2img(self.model.clap.audio_encoder.base.htsat, x)
            )
            self.model.clap.to(torch.device(self.device))

    def _move_to_device(self, x: torch.Tensor):
        if self.device == "cuda" and torch.cuda.is_available():
            return x.to(torch.device(self.device))
        elif self.device == "mps" and torch.backends.mps.is_available():
            return x.to(torch.device(self.device))
        return x

    def get_audio_embeddings(self, audio_files: list[str]) -> torch.Tensor:
        preprocessed_audio = self.model.preprocess_audio(audio_files, resample=True)
        preprocessed_audio = self._move_to_device(preprocessed_audio)  # type: ignore
        embeddings = self.model._get_audio_embeddings(preprocessed_audio)
        return embeddings

    def get_text_embeddings(self, text: list[str]) -> torch.Tensor:
        preprocessed_text = self.model.preprocess_text(text)
        for key in preprocessed_text:
            preprocessed_text[key] = self._move_to_device(preprocessed_text[key])
        embeddings = self.model._get_text_embeddings(preprocessed_text)
        return embeddings
