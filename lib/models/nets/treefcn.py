# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class RefineResidual(nn.Module):
    def __init__(self, in_planes, out_planes, ksize=3, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-5):
        super(RefineResidual, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                  stride=1, padding=0, dilation=1,
                                  bias=has_bias)
        self.cbr = ConvBnRelu(out_planes, out_planes, ksize, 1,
                              ksize // 2, has_bias=has_bias,
                              norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1x1(x)
        t = self.cbr(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x

class TreeFCN(nn.Module):
    def __init__(self, configer):
        super(TreeFCN, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        business_channel_num = self.configer.get('network', 'business_channel_num')
        embed_channel_num = self.configer.get('network', 'embed_channel_num')
        block_channel_nums = self.configer.get('network', 'block_channel_nums')
        tree_filter_group_num = self.configer.get('network', 'tree_filter_group_num')

        norm_layer = ModuleHelper.BatchNorm2d(self.configer.get('network', 'bn_type'))

        self.latent_layers = nn.ModuleList()
        self.refine_layers = nn.ModuleList()
        self.embed_layers = nn.ModuleList()
        self.mst_layers = nn.ModuleList()
        self.tree_filter_layers = nn.ModuleList()
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(block_channel_nums[-1], business_channel_num,
                       1, 1, 0, has_bn=True, has_relu=True, has_bias=False,
                       norm_layer=norm_layer))

        self.predict_layer = PredictHead(business_channel_num, self.num_classes, 4, norm_layer=norm_layer)
        for idx, channel in enumerate(block_channel_nums[::-1]):
            self.latent_layers.append(
                RefineResidual(channel, business_channel_num, 3,
                               norm_layer=norm_layer, has_relu=True)
            )
            self.refine_layers.append(
                RefineResidual(business_channel_num, business_channel_num, 3,
                               norm_layer=norm_layer, has_relu=True)
            )
            self.embed_layers.append(
                ConvBnRelu(business_channel_num, embed_channel_num, 1, 1, 0, has_bn=False,
                           has_relu=False, has_bias=False, norm_layer=norm_layer))
            self.mst_layers.append(MinimumSpanningTree(TreeFilter2D.norm2_distance))
            self.tree_filter_layers.append(TreeFilter2D(groups=tree_filter_group_num))

        # extra added layers for high-level MST
        self.embedding_layer = None
        if self.configer.get('tree_loss', 'params')['enable_high_level']:
            mid_channel = 256
            self.embedding_layer = ConvBnRelu(512, mid_channel, 1, 1, 0, has_bn=True, has_relu=True,
                                              has_bias=False, norm_layer=norm_layer)


    def forward(self, data, images=None, targets=None):
        batch, _, org_h, org_w = data.size()

        blocks = self.backbone(data)
        blocks.reverse()

        gap = self.global_context(blocks[0])
        scale_factor = (blocks[0].shape[2], blocks[0].shape[3])
        last_fm = F.interpolate(gap, scale_factor=scale_factor, mode='bilinear', align_corners=True)

        refined_fms = []
        for idx, (fm, latent_layer, refine_layer,
                  embed_layer, mst, tree_filter) in enumerate(
            zip(blocks,
                self.latent_layers,
                self.refine_layers,
                self.embed_layers,
                self.mst_layers,
                self.tree_filter_layers)):
            latent = latent_layer(fm)
            if last_fm is not None:
                tree = mst(fm)
                embed = embed_layer(last_fm)
                fusion = latent + tree_filter(last_fm.float(), embed.float(), tree, low_tree=False)
                refined_fms.append(refine_layer(fusion))
            else:
                refined_fms.append(latent)
            last_fm = F.interpolate(refined_fms[-1], size=blocks[idx+1].size()[2:], mode='bilinear', align_corners=True)

        pred = self.predict_layer(refined_fms[-1])
        outs = []
        outs.append(F.interpolate(pred, size=(org_h, org_w), mode="bilinear", align_corners=True))

        if self.training:
            if not self.embedding_layer:
                embed_feat = None
            else:
                embed_feat = self.embedding_layer(refined_fms[-1])
            outs.append(pred)
            outs.append(embed_feat)

        return outs


class PredictHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale, norm_layer=nn.BatchNorm2d):
        super(PredictHead, self).__init__()
        self.head_layers = nn.Sequential(
            RefineResidual(in_planes, in_planes, norm_layer=norm_layer, has_relu=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0))

        self.scale = scale

    def forward(self, x):
        x = self.head_layers(x)
        # x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return x