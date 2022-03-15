import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.decoder_block import DeepLabV3PlusHead, ASPPModule
from lib.models.tools.module_helper import ModuleHelper


class DeepLabV3Plus(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3Plus, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        in_channels = 2048
        stride = self.configer.get('network', 'stride')
        if stride == 8:
            d_rate=[12, 24, 36]
            skip_4x_ch = 256
        else:
            d_rate = [6, 12, 18]
            skip_4x_ch = 256

        self.aspp = ASPPModule(in_channels, 256, d_rate=d_rate, bn_type=self.configer.get('network', 'bn_type'))

        self.decoder = DeepLabV3PlusHead(num_classes=self.num_classes, bottleneck_ch=256, skip_4x_ch=skip_4x_ch, bn_type=self.configer.get('network', 'bn_type'))

        # extra added layers for high-level MST
        self.embedding_layer = None
        if self.configer.get('tree_loss', 'params')['enable_high_level']:
            mid_channel = 256
            self.embedding_layer = nn.Sequential(
                nn.Conv2d(256, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                ModuleHelper.BNReLU(mid_channel, bn_type=self.configer.get('network', 'bn_type'))
            )

    def forward(self, x):
        b, _, org_h, org_w = x.size()
        x = self.backbone(x)
        high_feat = self.aspp(x[-1])
        low_feat = x[2]

        main_pred, feat = self.decoder(high_feat, low_feat)
        outs = []
        outs.append(F.interpolate(main_pred, size=(org_h, org_w), mode='bilinear', align_corners=True))

        if self.training:
            if not self.embedding_layer:
                embed_feat = None
            else:
                embed_feat = self.embedding_layer(feat)
            outs.append(main_pred)
            outs.append(embed_feat)
        return outs
