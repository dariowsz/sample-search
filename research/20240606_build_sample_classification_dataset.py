# %%
import research.start  # noqa isort:skip

# %%
import os
from glob import glob

import librosa
from torch.utils.data import Dataset


# %%
def _transform(sample_path: str):
    audio_data, _ = librosa.load(sample_path, sr=48000)
    return audio_data.reshape(1, -1)


class SamplesDataset(Dataset):
    def __init__(self, sample_dirs: list[str], transform=None):
        self.sample_paths = []
        for sample_dir in sample_dirs:
            self.sample_paths += glob(
                os.path.join(sample_dir, "**/*.wav"), recursive=True
            )
        self.transform = transform

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.sample_paths[idx])
        return self.sample_paths[idx]


# %%
sample_dirs = [
    "/Users/dario.wisznewer/Cymatics",
    "/Users/dario.wisznewer/Samples",
    "/Users/dario.wisznewer/Splice",
]
dataset = SamplesDataset(sample_dirs)

# %%
dataset[0]

# %%
