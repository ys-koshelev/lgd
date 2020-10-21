import torch.nn as nn

from forked.mit_semseg.models import ModelBuilder, SegmentationModule
from .base import NetworkDegradationBase
from scipy.io import loadmat
from forked.mit_semseg.utils import colorEncode
import torch as th


class SegmentationDegradation(NetworkDegradationBase):
    """
    This class represents semantic segmentation degradation, approximated by convolution neural network (CNN)
    """
    def __init__(self, device: str = 'cpu') -> None:
        segmentation_network = SegmentationModule(ModelBuilder.build_encoder('resnet18dilated'),
                                                  ModelBuilder.build_decoder('ppm_deepsup'))
        segmentation_network.encoder.load_state_dict(th.load('pretrained/encoder_epoch_20.pth', map_location='cpu'))
        segmentation_network.decoder.load_state_dict(th.load('pretrained/decoder_epoch_20.pth', map_location='cpu'))
        super().__init__(segmentation_network, nn.CrossEntropyLoss(), device=device)
        self.colors = loadmat('pretrained/color150.mat')['colors']

    def init_latent_images(self, labels: th.Tensor) -> th.Tensor:
        """
        This method is used to init latent images from degraded ones for the first restoration step.

        :param labels: batch of labels of shape [B, H, W] to create image
        :return: initialized latent images of shape [B, 3, H, W]
        """
        images_batch = []
        for label in labels:
            images_batch.append(th.from_numpy(colorEncode(label.detach().cpu().numpy(), self.colors, mode='RGB')))
        images_batch = th.stack(images_batch, dim=0).to(device=labels.device).permute(0, 3, 1, 2).contiguous().float()
        return images_batch/255
