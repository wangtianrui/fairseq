from email.policy import default
import math
from turtle import forward

from importlib_metadata import metadata
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.logging.meters import safe_round
from fairseq.utils import is_xla_tensor


@dataclass
class SECriterionConfig(FairseqDataclass):
    loss_name: str = field(
        default="si_snr", metadata={"help": "loss name: si_snr, ..."}
    )


def l2_norm(s1, s2):
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, training=False, eps=1e-8):
    # s1: est , s2: clean
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return -torch.mean(snr)


@register_criterion("speechenhancement_loss", dataclass=SECriterionConfig)
class SpeechEnhancementCriterion(FairseqCriterion):
    def __init__(self, task, loss_name):
        super().__init__(task)
        self.loss_name = loss_name

    def forward(self, model, sample, reduce=True):
        """
        Args:
            model (_type_): _description_
            sample (_type_): {"net_input": {"noisy": noisys}, "target": targets}
            reduce (bool, optional): _description_. Defaults to True.
        """
        net_output = model(**sample["net_input"])
        if self.loss_name == "si_snr":
            loss = si_snr(net_output, sample["target"])
            
        logging_output = {
            "si_snr_loss": loss.item()
        }
        
        sample_size = sample["target"].numel()
        return loss, sample_size, logging_output