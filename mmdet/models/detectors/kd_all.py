# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .gfl import GFL

import torch
from torch import nn
import torch.nn.functional as F
from mmdet.models import build_detector
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint

from torchvision.ops import DeformConv2d


@DETECTORS.register_module()
class KD_GFL_ALL(GFL):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,

                 tea_model,
                 tea_pretrained=None,

                 # idea
                 imitation_loss_weigth=0.1,
                 imitation_fuse_loss_weigth=0.1,
                 mul_fuse=True,

                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None
                 ):
        super(KD_GFL_ALL, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)
        self.tea_model = build_detector(tea_model, train_cfg=train_cfg, test_cfg=test_cfg)
        self.load_weights(pretrained=tea_pretrained)
        self.freeze_models()
        self.imitation_loss_weigth = imitation_loss_weigth
        self.imitation_fuse_loss_weigth = imitation_fuse_loss_weigth
        self.kd_trans = build_kd_trans(self.imitation_loss_weigth)
        self.mul_fuse = mul_fuse
        self.ccd = CCD(mul_fuse=self.mul_fuse)

 
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        self.tea_model.eval()
        with torch.no_grad():
            tea_x = self.tea_model.extract_feat(img)

        stu_x = self.extract_feat(img) 

        with torch.no_grad():
            teacher_x = self.tea_model.extract_feat(img)
            out_teacher = self.tea_model.bbox_head(teacher_x)

        losses = self.bbox_head.forward_train(stu_x, out_teacher, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)

        s_features = self.kd_trans(stu_x)
        t_features = list(tea_x)

        kd_loss, kd_loss_fuse = self.ccd(stu_x, s_features, t_features)
        kd_loss = kd_loss * self.imitation_loss_weigth
        kd_loss_fuse = kd_loss_fuse * self.imitation_fuse_loss_weigth

        losses.update(dict(loss_att=kd_loss))
        losses.update(dict(loss_CCD=kd_loss_fuse))
        return losses

    def freeze_models(self):
        self.tea_model.eval()
        for param in self.tea_model.parameters():
            param.requires_grad = False

    def load_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self.tea_model, pretrained, strict=False, logger=logger)
            print("load teacher model success from {}".format(self.tea_model))
        else:
            raise TypeError('pretrained must be a str')

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.tea_model.cuda(device=device)
        return super().cuda(device=device)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'tea_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

class LIC(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(LIC, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out

class CF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(CF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            LIC(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None


    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            shape = x.shape[-2:]
            y = F.interpolate(y, shape, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output
        y = self.conv2(x)
        return y, x

class Trans(nn.Module):
    def __init__(
        self, in_channels, out_channels, mid_channel
    ):
        super(Trans, self).__init__()

        cfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            cfs.append(CF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))

        self.cfs = cfs[::-1]

    def forward(self, student_features):
        x = student_features[::-1]
        results = []
        out_features, res_features = self.cfs[0](x[0])
        results.append(out_features)
        for features, cf in zip(x[1:], self.cfs[1:]):
            out_features, res_features = cf(features, res_features)
            results.insert(0, out_features)

        return results

def build_kd_trans(cfg):
    in_channels = [256,256,256,256,256]
    out_channels = [256,256,256,256,256]
    mid_channel = 256
    model = Trans(in_channels, out_channels, mid_channel)
    return model

class CCD(nn.Module):
    def __init__(self, mul_fuse=True):
        super(CCD, self).__init__()
        self.mul_fuse = mul_fuse

        self.att_adaptive_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                                  nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                                  nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                                  nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                                                  nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)])

    def mask_att(self, x, temperature):
        spatial_att = torch.mean(torch.abs(x), dim=1, keepdim=True)  # (N,1,H,W)
        s_size = spatial_att.size()
        spatial_att = spatial_att.view(x.size(0), -1)
        spatial_att = torch.softmax(spatial_att / temperature, dim=1) * s_size[2] * s_size[3]
        spatial_att = spatial_att.view(s_size)
        return spatial_att

    def forward(self, fstudent, fstudent_fuse, fteacher):
        loss, loss_fuse = 0, 0
        for i in range(len(fstudent)):
            fs, ft = fstudent[i], fteacher[i]
            fs_fuse = fstudent_fuse[i]
            n, c, h, w = fs.shape

            # same scale
            stu_spatial_att = self.mask_att(x=fs, temperature=0.1)
            tea_spatial_att = self.mask_att(x=ft, temperature=0.1)
            sum_spatial_att = (stu_spatial_att + tea_spatial_att) / 2
            sum_spatial_att = sum_spatial_att.detach()
            tmp = (self.att_adaptive_layers[i](fs) - ft) ** 2 * sum_spatial_att
            loss += torch.sum(tmp) ** 0.5

            # fused scale
            loss_fuse += F.mse_loss(fs_fuse, ft, reduction='mean')

            if self.mul_fuse == True:
                loss_mul = 0
                cnt = 1.0
                tot = 1.0
                for l in [4, 2, 1]:
                    if l >= h:
                        continue
                    tmpfs = F.adaptive_avg_pool2d(fs, (l, l)) # to do!
                    tmpft = F.adaptive_avg_pool2d(ft, (l, l))
                    cnt /= 2.0
                    loss_mul += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                    tot += cnt
                loss_mul = loss_mul / tot
                loss_fuse += loss_mul

        return loss, loss_fuse