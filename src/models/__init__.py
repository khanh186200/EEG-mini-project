"""Các kiến trúc mô hình hiện có cho phân loại single-trial EEG P300."""

from src.models.ms_cnn_bigru import MSCNNBiGRU
from src.models.p300_atcnet import P300ATCNet

__all__ = ["MSCNNBiGRU", "P300ATCNet"]
