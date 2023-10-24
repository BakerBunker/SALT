from pathlib import Path
import random

import torch
from torch import Tensor
import torchaudio
import pandas as pd

from matcher import KNeighborsVC


class Anonymizer:
    def __init__(self, knnvc: KNeighborsVC) -> None:
        self.knnvc = knnvc
        self.pool = {}
        self.fake_speakers = {}

    def add_speaker(
        self,
        name: str,
        wavs: list[Path] | list[Tensor] | None = None,
        preprocessed_file: Path | None = None,
    ):
        if preprocessed_file:
            self.pool[name] = torch.load(preprocessed_file)
        elif wavs:
            sample_wav, sr = torchaudio.load(wavs[0])
            self.pool[name] = (self.knnvc.get_matching_set(wavs), sample_wav, sr)
        else:
            raise ValueError("No valid input")

    def interpolate(
        self,
        wav,
        speaker_dict: pd.DataFrame | dict,
        topk: int = 4,
    ):
        if speaker_dict is dict:
            speaker_dict = pd.DataFrame.from_dict(
                {
                    "speaker": list(speaker_dict.keys()),
                    "weight": list(speaker_dict.values()),
                }
            )
        src_feat = self.knnvc.get_features(wav)
        tgt = torch.zeros_like(src_feat)
        for _, name, weight in speaker_dict.itertuples():
            if name == "self":
                tgt += src_feat * weight
            else:
                tgt += (
                    self.knnvc.match_feat(
                        src_feat, self.pool[name][0], topk=topk
                    ).squeeze(0)
                    * weight
                )
        wav = self.knnvc.feat_to_wav(tgt.unsqueeze(0))
        return wav

    def build_fake_speaker(self, speaker_dict: pd.DataFrame, extrapolation_factor=0):
        def process(row):
            row.weight = row.weight * (
                extrapolation_factor + 1
            ) - extrapolation_factor / len(speaker_dict)
            return row

        return speaker_dict.apply(process, axis=1)

    def add_fake_speaker(
        self,
        name: str,
        speaker_dict: dict,
    ):
        self.fake_speakers[name] = speaker_dict

    def make_speaker_pack(self, wavs, name):
        mset = self.knnvc.get_matching_set(wavs)
        wav, sr = torchaudio.load(wavs[0])
        p=f"assets/{name}.pack"
        torch.save((mset, wav, sr), p)
        print(f'Speaker pack saved to {p}')
        return p

    def get_random_speaker(self, speakers=4, preservation_factor=0):
        assert speakers > 0 and 0 <= preservation_factor < 1
        speakers = random.sample(list(self.pool.keys()), k=speakers)
        weights = (
            (torch.randn(len(speakers)).softmax(-1) * (1 - preservation_factor))
            .cpu()
            .tolist()
        )
        if preservation_factor != 0:
            speakers.append("self")
            weights.append(preservation_factor)
        df = pd.DataFrame.from_dict(
            {
                "speaker": speakers,
                "weight": weights,
            }
        )
        return df
