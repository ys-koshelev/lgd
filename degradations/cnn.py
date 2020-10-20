import torch.nn as nn

from forked.mit_semseg.models import ModelBuilder, SegmentationModule
from .base import NetworkDegradationBase


class SegmentationDegradation(NetworkDegradationBase):
    """
    This class represents semantic segmentation degradation, approximated by neural network (NN)
    """
    def __init__(self, device: str = 'cpu') -> None:
        segmentation_network = SegmentationModule(ModelBuilder.build_encoder('resnet18dilated'),
                                                  ModelBuilder.build_decoder('ppm_deepsup'), nn.CrossEntropyLoss)
        super().__init__(segmentation_network, nn.CrossEntropyLoss, device=device)
