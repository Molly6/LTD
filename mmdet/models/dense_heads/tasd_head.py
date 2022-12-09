# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.core import (bbox_overlaps, multi_apply, reduce_mean,
                        images_to_levels, anchor_inside_flags, unmap)
from ..builder import HEADS, build_loss
from .gfl_head import GFLHead

from torchvision import transforms
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import os



@HEADS.register_module()
class TASDHead(GFLHead):
    """Localization distillation Head. (Short description)

    It utilizes the learned bbox distributions to transfer the localization
    dark knowledge from teacher to student. Original paper: `Localization
    Distillation for Object Detection. <https://arxiv.org/abs/2102.12252>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss_ld (dict): Config of Localization Distillation Loss (LD),
            T is the temperature for distillation.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_ld=dict(
                     type='LocalizationDistillationLoss',
                     loss_weight=0.25,
                     T=10),
                 loss_ld_vlr=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.25,
                     T=10),
                 loss_kd=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=10,
                     T=2),
                 loss_kd_vlr=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=10,
                     T=2),
                 **kwargs):

        super(TASDHead, self).__init__(num_classes, in_channels, **kwargs)
        self.loss_ld = build_loss(loss_ld)
        self.loss_ld_vlr = build_loss(loss_ld_vlr)
        self.loss_kd = build_loss(loss_kd)
        self.loss_kd_vlr = build_loss(loss_kd_vlr)



    def forward_train(self,
                      x,
                      out_teacher,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        # soft_target = out_teacher[1]
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, out_teacher, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, out_teacher, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, stride, soft_targets, soft_label,
                    vlr_region, vlr_region_cls, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[tuple, Tensor]: Loss components and weight targets.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'


        H, W = bbox_pred.size(2), bbox_pred.size(3)

        anchors = anchors.reshape(-1, 4)

        cls_score = cls_score.permute(0, 2, 3,  # (N*H*W, num_classes)
                                      1).reshape(-1, self.cls_out_channels)

        bbox_pred = bbox_pred.permute(0, 2, 3,  # (N*H*W, 4*(n+1))
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        soft_targets = soft_targets.permute(0, 2, 3,  # (N*H*W, 4*(n+1))
                                            1).reshape(-1,
                                                       4 * (self.reg_max + 1))
        soft_label = soft_label.permute(0, 2, 3,
                                        1).reshape(-1, self.cls_out_channels)

        bbox_targets = bbox_targets.reshape(-1, 4)  # (N*num_total_anchors, 4)
        labels = labels.reshape(-1)  # (N*num_total_anchors)
        label_weights = label_weights.reshape(-1)

        vlr_region = vlr_region.reshape(-1)
        vlr_region_cls = vlr_region_cls.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)  # anchor中正样本的索引
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)



        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            pos_soft_targets = soft_targets[pos_inds]
            soft_corners = pos_soft_targets.reshape(-1, self.reg_max + 1)

            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

            # ld loss
            # loss_ld = bbox_pred.sum() * 0
            ld_weight = torch.softmax(soft_corners.sum(dim=1)/5, dim=0)*soft_corners.size(0)
            loss_ld = self.loss_ld(
                pred_corners,
                soft_corners,
                weight=ld_weight,#weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=16) #4.0,pred_corners.shape[0]
            # print("ld_weight", ld_weight)
            # print("w",weight_targets[:, None].expand(-1, 4).reshape(-1).shape)

            # kd loss #########
            # loss_kd = bbox_pred.sum() * 0
            # kd_weight = torch.softmax(soft_label[pos_inds].max(dim=1)[0] / 5, dim=0) * soft_label[pos_inds].size(0)
            loss_kd = self.loss_kd(
                cls_score[pos_inds],
                soft_label[pos_inds],
                weight=None, #label_weights[pos_inds]
                avg_factor=16) #pos_inds.shape[0]
            # print("soft_label", soft_label.sum(dim=1))

        else:
            loss_ld = bbox_pred.sum() * 0
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_kd = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # ld_vlr ########
        remain_inds = (vlr_region > 0).nonzero().squeeze(1)



        remain_inds_cls = (vlr_region > 0).nonzero().squeeze(1)
        # remain_inds_cls = (vlr_region_cls > 0).nonzero().squeeze(1)
        if len(remain_inds) > 0:
            neg_pred_corners = bbox_pred[remain_inds].reshape(
                -1, self.reg_max + 1)
            neg_soft_corners = soft_targets[remain_inds].reshape(
                -1, self.reg_max + 1)
            # print("neg_soft_corners", neg_soft_corners.shape)

            remain_targets = vlr_region[remain_inds]

            ld_vlr_weight = torch.softmax(neg_soft_corners.sum(dim=1)/5, dim=0)*neg_soft_corners.size(0)
            loss_ld_vlr = self.loss_ld_vlr(
                neg_pred_corners,
                neg_soft_corners,
                weight=ld_vlr_weight, #remain_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=16.0) # neg_pred_corners.shape[0]

            ### 尝试cls
            # kd_vlr_weight = torch.softmax(soft_label[remain_inds_cls].max(dim=1)[0] / 10, dim=0) * soft_label[remain_inds_cls].size(0)
            loss_kd_vlr = self.loss_kd_vlr(
                cls_score[remain_inds_cls],
                soft_label[remain_inds_cls],
                weight=None, #remain_targets
                avg_factor=16) #remain_inds.shape[0]

        else:
            loss_ld_vlr = bbox_pred.sum() * 0
            loss_kd_vlr = bbox_pred.sum() * 0

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, loss_ld, loss_ld_vlr, loss_kd, loss_kd_vlr, weight_targets.sum(),

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             soft_teacher,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        soft_label, soft_target = soft_teacher

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg, assigned_neg_list, vlr_regions_cls_list) = cls_reg_targets

        ## reg,cls可视化
        plt.figure(figsize=(25, 25))
        file_name = os.path.basename(os.path.splitext(img_metas[0]['filename'])[0])
        plt.title(file_name)
        bg_class_ind = self.num_classes
        for i in range(len(bbox_targets_list)):
            # anchor中正样本的索引
            labels = labels_list[i]
            labels = labels.reshape(-1)

            pos_inds = ((labels >= 0)
                        & (labels < bg_class_ind)).nonzero().squeeze(1)

            # reg
            vlr_region = assigned_neg_list[i]
            vlr_region = vlr_region.reshape(-1)
            remain_inds = (vlr_region > 0).nonzero().squeeze(1)

            bbox_pred = bbox_preds[i]
            H, W = bbox_pred.size(2), bbox_pred.size(3)
            bbox_pred = bbox_pred.permute(0, 2, 3,  # (N*H*W, 4*(n+1))
                                          1).reshape(-1, 4 * (self.reg_max + 1))

            reg_map = torch.zeros(bbox_pred.size(0))
            reg_map[pos_inds] = 1
            reg_map[remain_inds] = 1
            reg_map = torch.unsqueeze(reg_map.reshape(H, W), 2)
            img_reg = reg_map.cpu().clone()

            plt.subplot(3, 5, i+1)
            plt.imshow(img_reg)  # cmap='jet'

            # cls
            vlr_region_cls = vlr_regions_cls_list[i]
            vlr_region_cls = vlr_region_cls.reshape(-1)
            remain_inds_cls = (vlr_region_cls > 0).nonzero().squeeze(1)

            cls_map = torch.zeros(bbox_pred.size(0))
            cls_map[pos_inds] = 1
            cls_map[remain_inds_cls] = 1
            cls_map = torch.unsqueeze(cls_map.reshape(H, W), 2)
            img_cls = cls_map.cpu().clone()

            plt.subplot(3, 5, i+6)
            plt.imshow(img_cls)  # cmap='jet'

        image = plt.imread(img_metas[0]['filename'])
        plt.subplot(3, 5, 11)
        plt.imshow(image)
        # plt.show()
        fig_path = '/workspace/code/mmdet2_19/visual/' + file_name + '.png'
        plt.savefig(fig_path, dpi=1000, alpha=0.4)





        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_dfl, losses_ld, losses_ld_vlr, loss_kd, loss_kd_vlr,\
            avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.prior_generator.strides,
                soft_target,
                soft_label,
                assigned_neg_list,
                vlr_regions_cls_list,
                num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor) + 1e-6
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = [x / avg_factor for x in losses_bbox]
        losses_dfl = [x / avg_factor for x in losses_dfl]
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_ld=losses_ld,
            loss_ld_vlr=losses_ld_vlr,
            loss_kd=loss_kd,
            loss_kd_vlr=loss_kd_vlr,
        )

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list, all_vlr_region, all_vlr_region_cls) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        #num_total_remain_neg = sum([max(inds.numel(), 1) for inds in assigned_neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        vlr_regions_list = images_to_levels(all_vlr_region, num_level_anchors)
        vlr_regions_cls_list = images_to_levels(all_vlr_region_cls, num_level_anchors)
        # sampled anchors of all images

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg, vlr_regions_list, vlr_regions_cls_list)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        # Check whether the anchors are inside the border.
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        vlr_region, vlr_region_cls = self.assigner.get_vlr_region(anchors, # (N, 4)
                                                  num_level_anchors_inside, #
                                                  gt_bboxes, # (k, 4)
                                                  gt_bboxes_ignore,
                                                  gt_labels) # (k, )



        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)

        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)


        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            vlr_region = unmap(vlr_region, num_total_anchors, inside_flags)
            vlr_region_cls = unmap(vlr_region_cls, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, vlr_region, vlr_region_cls)
