
from email.policy import default
import json
import logging
import math
from argparse import Namespace
from pathlib import Path

import numpy as np
from numpy import dtype
from dataclasses import dataclass, field
from fairseq.data.fairseq_dataset import FairseqDataset
from omegaconf import MISSING, II, OmegaConf

import torch
import os
import torch.nn as nn
import soundfile as sf

from fairseq import utils
from fairseq.data import Dictionary
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from . import FairseqTask, register_task

logger = logging.getLogger(__name__)


@dataclass
class SpeechEnhancementConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    sample_dur: float = field(default=5.0, metadata={"help": "audio during in secends"})


class PairEnhancementDataset(FairseqDataset):
    def __init__(self, data_home, data_names, sample_rate, sample_dur, shuffle=True):
        super().__init__()
        self.data_home = data_home
        self.data_names = data_names  # [ [noisy/xxx.wav, clean/xxx.wav], ... ]
        self.sample_rate = sample_rate
        self.sample_dur = sample_dur
        self.shuffle = shuffle
        self.sample_size = int(self.sample_rate * self.sample_dur)

    def __len__(self):
        return len(self.data_names)

    def num_tokens(self, index):
        # 用来告诉dataloader每个句子有多长
        return int(self.sample_rate * self.sample_dur)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.data_names),
                    int(self.sample_rate * self.sample_dur),
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def __getitem__(self, index):
        noisy_path, clean_path = self.data_names[index]
        noisy_wave, noisy_sr = sf.read(os.path.join(self.data_home, noisy_path), dtype="float32")
        clean_wave, clean_sr = sf.read(os.path.join(self.data_home, clean_path), dtype="float32")

        assert noisy_sr == clean_sr and noisy_sr == self.sample_rate, "sample rate error"

        if len(clean_wave) < len(noisy_wave):
            clean_wave = np.pad(clean_wave, [0, len(noisy_wave)-len(clean_wave)], 1e-8)
        elif len(clean_wave) > len(noisy_wave):
            clean_wave = clean_wave[:len(noisy_wave)]

        noisy_wave = torch.from_numpy(noisy_wave).float()
        clean_wave = torch.from_numpy(clean_wave).float()
        return {"id": index, "source": noisy_wave, "target": clean_wave}

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        # ids = torch.LongTensor([s["id"] for s in samples])
        sources = torch.stack([s["source"] for s in samples], dim=0)
        targets = torch.stack([s["target"] for s in samples], dim=0)
        net_inp = {"noisy": sources}
        return {"net_input": net_inp, "target": targets}


@register_task("speech_enhancement", dataclass=SpeechEnhancementConfig)
class SpeechEnhancementTask(FairseqTask):
    @classmethod
    def setup_task(cls, cfg: SpeechEnhancementConfig, **kwargs):
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs):
        data_path = self.cfg.data
        manifest_path = os.path.join(data_path, "{}.tsv".format(split))
        data_names = []
        with open(manifest_path, "r") as f:
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2, line
                data_names.append([items[0],  items[1]])  # noisy, clean
        data_names = np.array(data_names)
        self.datasets[split] = PairEnhancementDataset(
            data_home=data_path, data_names=data_names, sample_rate=self.cfg.sample_rate, sample_dur=self.cfg.sample_dur)
        logging.info("%s data is loaded, len: %d" % (split, len(data_names)))

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None
