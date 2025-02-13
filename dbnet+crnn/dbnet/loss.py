import torch
import torch.nn as nn
import torch.nn.functional


class BalanceCrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.negative_ratio = 3.0
        self.eps = 1e-6

    def forward(self, inputs):

        prob_pred, prob_map, prob_mask, prob_weight = inputs
        # 二元交叉熵
        # loss = nn.functional.binary_cross_entropy(input=prob_pred, target=prob_map, reduction='none')
        loss = nn.functional.binary_cross_entropy(input=prob_pred, target=prob_map, weight=prob_weight, reduction='none')

        positive_area = prob_map * prob_mask
        negative_area = (1 - prob_map) * prob_mask

        positive_count = int((positive_area > 0.5).sum())
        negative_count = min(int((negative_area > 0.5).sum()), int(positive_count * self.negative_ratio))

        positive_loss = loss * positive_area
        negative_loss = loss * negative_area

        negative_loss, _ = negative_loss.view(-1).contiguous().topk(negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        return balance_loss


class MaskL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, inputs):
        thresh_pred, thresh_map, thresh_mask = inputs
        # loss = (torch.abs(thresh_pred - thresh_map) * thresh_mask).sum() / (thresh_mask.sum() + self.eps)
        # loss = (loss * thresh_mask).sum() / (thresh_mask.sum() + self.eps)
        loss = torch.nn.functional.smooth_l1_loss(thresh_pred * thresh_mask, thresh_map * thresh_mask, reduction='mean')
        return loss


class DiceLoss(nn.Module):

    # Loss function from https://arxiv.org/abs/1707.03237,
    # where iou computation is introduced heatmap manner to measure the
    # diversity bwtween tow heatmaps.
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, inputs):
        binary_pred, prob_map, prob_mask = inputs

        assert (len(prob_map.shape) == 4)  # gt.shape == [batch, 1, height, width]
        assert binary_pred.shape == prob_map.shape
        assert binary_pred.shape == prob_mask.shape
        # assert binary_pred.shape == prob_weight.shape

        # prob_mask = prob_mask * prob_weight
        # 求dice损失，和交并比iou相似，2倍交集除以元素和(交集+并集)
        inter = (binary_pred * prob_map * prob_mask).sum()
        union = (binary_pred * prob_mask).sum() + (prob_map * prob_mask).sum() + self.eps

        loss = 1 - 2.0 * inter / union
        assert loss <= 1
        return loss


class DBLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.k = 50
        self.alpha = 1.0
        self.beta = 10.0
        self.gama = 1.0

        self.bce_loss = BalanceCrossEntropyLoss()
        self.maskl1_loss = MaskL1Loss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs):

        preds, features = inputs

        prob_pred, thresh_pred = preds
        prob_map, prob_mask, prob_weight, thresh_map, thresh_mask = features
        # 近似二值图生成：
        binary_pred = torch.reciprocal(1.0 + torch.exp(-self.k * (prob_pred - thresh_pred)))

        # 概率图与标签概率图求损失
        prob_inputs = [prob_pred, prob_map, prob_mask, prob_weight]
        # 阈值图与标签阈值图求损失
        thresh_inputs = [thresh_pred, thresh_map, thresh_mask]
        # 生成近似二值图 与标签概率图求损失
        binary_inputs = [binary_pred, prob_map, prob_mask]

        prob_loss = self.bce_loss(prob_inputs)
        thresh_loss = self.maskl1_loss(thresh_inputs)
        binary_loss = self.dice_loss(binary_inputs)

        loss = self.alpha * prob_loss + self.beta * thresh_loss + self.gama * binary_loss
        return loss
