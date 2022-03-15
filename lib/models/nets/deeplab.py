import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.decoder_block import DeepLabHead, DeepLabV2Head
from lib.models.tools.module_helper import ModuleHelper
from lib.utils.tools.logger import Logger as Log

class DeepLabV3(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        self.decoder = DeepLabHead(num_classes=self.num_classes, bn_type=self.configer.get('network', 'bn_type'))


    def forward(self, x_, images=None, targets=None):
        _, _, org_h, org_w = x_.size()

        x = self.backbone(x_)
        main_pred, feat = self.decoder(x)

        outs = []
        outs.append(F.interpolate(main_pred, size=(org_h, org_w), mode="bilinear", align_corners=True))

        return outs