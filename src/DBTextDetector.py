#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : DBTextDetector.py.py
# @Author   : jade
# @Date     : 2021/9/1 9:27
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :

from paddle import fluid
from copy import deepcopy
from paddle.fluid.param_attr import ParamAttr
import math
import numpy as np
from shapely.geometry import Polygon
import pyclipper

def BalanceLoss(pred,
                gt,
                mask,
                balance_loss=True,
                main_loss_type="DiceLoss",
                negative_ratio=3,
                return_origin=False,
                eps=1e-6):
    """
    The BalanceLoss for Differentiable Binarization text detection
    args:
        pred (variable): predicted feature maps.
        gt (variable): ground truth feature maps.
        mask (variable): masked maps.
        balance_loss (bool): whether balance loss or not, default is True
        main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
            'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
        negative_ratio (int|float): float, default is 3.
        return_origin (bool): whether return unbalanced loss or not, default is False.
        eps (float): default is 1e-6.
    return: (variable) balanced loss
    """
    positive = gt * mask
    negative = (1 - gt) * mask

    positive_count = fluid.layers.reduce_sum(positive)
    positive_count_int = fluid.layers.cast(positive_count, dtype=np.int32)
    negative_count = min(
        fluid.layers.reduce_sum(negative), positive_count * negative_ratio)
    negative_count_int = fluid.layers.cast(negative_count, dtype=np.int32)

    if main_loss_type == "CrossEntropy":
        loss = fluid.layers.cross_entropy(input=pred, label=gt, soft_label=True)
        loss = fluid.layers.reduce_mean(loss)
    elif main_loss_type == "Euclidean":
        loss = fluid.layers.square(pred - gt)
        loss = fluid.layers.reduce_mean(loss)
    elif main_loss_type == "DiceLoss":
        loss = DiceLoss(pred, gt, mask)
    elif main_loss_type == "BCELoss":
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(pred, label=gt)
    elif main_loss_type == "MaskL1Loss":
        loss = MaskL1Loss(pred, gt, mask)
    else:
        loss_type = [
            'CrossEntropy', 'DiceLoss', 'Euclidean', 'BCELoss', 'MaskL1Loss'
        ]
        raise Exception("main_loss_type in BalanceLoss() can only be one of {}".
                        format(loss_type))

    if not balance_loss:
        return loss

    positive_loss = positive * loss
    negative_loss = negative * loss
    negative_loss = fluid.layers.reshape(negative_loss, shape=[-1])
    negative_loss, _ = fluid.layers.topk(negative_loss, k=negative_count_int)
    balance_loss = (fluid.layers.reduce_sum(positive_loss) +
                    fluid.layers.reduce_sum(negative_loss)) / (
                        positive_count + negative_count + eps)

    if return_origin:
        return balance_loss, loss
    return balance_loss


def DiceLoss(pred, gt, mask, weights=None, eps=1e-6):
    """
    DiceLoss function.
    """

    assert pred.shape == gt.shape
    assert pred.shape == mask.shape
    if weights is not None:
        assert weights.shape == mask.shape
        mask = weights * mask
    intersection = fluid.layers.reduce_sum(pred * gt * mask)

    union = fluid.layers.reduce_sum(pred * mask) + fluid.layers.reduce_sum(
        gt * mask) + eps
    loss = 1 - 2.0 * intersection / union
    assert loss <= 1
    return loss


def MaskL1Loss(pred, gt, mask, eps=1e-6):
    """
    Mask L1 Loss
    """
    loss = fluid.layers.reduce_sum((fluid.layers.abs(pred - gt) * mask)) / (
        fluid.layers.reduce_sum(mask) + eps)
    loss = fluid.layers.reduce_mean(loss)
    return loss


class DBLoss(object):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    def __init__(self, balance_loss=True,main_loss_type="DiceLoss",alpha=5,beta=10,ohem_ratio=3):
        super(DBLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type

        self.alpha = alpha
        self.beta = beta
        self.ohem_ratio = ohem_ratio

    def __call__(self, predicts, labels):
        label_shrink_map = labels['shrink_map']
        label_shrink_mask = labels['shrink_mask']
        label_threshold_map = labels['threshold_map']
        label_threshold_mask = labels['threshold_mask']
        pred = predicts['maps']
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]

        loss_shrink_maps = BalanceLoss(
            shrink_maps,
            label_shrink_map,
            label_shrink_mask,
            balance_loss=self.balance_loss,
            main_loss_type=self.main_loss_type,
            negative_ratio=self.ohem_ratio)
        loss_threshold_maps = MaskL1Loss(threshold_maps, label_threshold_map,
                                         label_threshold_mask)
        loss_binary_maps = DiceLoss(binary_maps, label_shrink_map,
                                    label_shrink_mask)
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps

        loss_all = loss_shrink_maps + loss_threshold_maps\
            + loss_binary_maps
        losses = {'total_loss':loss_all,\
            "loss_shrink_maps":loss_shrink_maps,\
            "loss_threshold_maps":loss_threshold_maps,\
            "loss_binary_maps":loss_binary_maps}
        return losses


class DBHead(object):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self,image_shape,k=50,inner_channels=96):
        self.k = k
        self.inner_channels = inner_channels
        self.C, self.H, self.W = image_shape

    def binarize(self, x):
        conv1 = fluid.layers.conv2d(
            input=x,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=False)
        conv_bn1 = fluid.layers.batch_norm(
            input=conv1,
            param_attr=fluid.initializer.ConstantInitializer(value=1.0),
            bias_attr=fluid.initializer.ConstantInitializer(value=1e-4),
            act="relu")
        conv2 = fluid.layers.conv2d_transpose(
            input=conv_bn1,
            num_filters=self.inner_channels // 4,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=self._get_bias_attr(0.0004, conv_bn1.shape[1], "conv2"),
            act=None)
        conv_bn2 = fluid.layers.batch_norm(
            input=conv2,
            param_attr=fluid.initializer.ConstantInitializer(value=1.0),
            bias_attr=fluid.initializer.ConstantInitializer(value=1e-4),
            act="relu")
        conv3 = fluid.layers.conv2d_transpose(
            input=conv_bn2,
            num_filters=1,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=self._get_bias_attr(0.0004, conv_bn2.shape[1], "conv3"),
            act=None)
        out = fluid.layers.sigmoid(conv3)
        return out

    def thresh(self, x):
        conv1 = fluid.layers.conv2d(
            input=x,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=False)
        conv_bn1 = fluid.layers.batch_norm(
            input=conv1,
            param_attr=fluid.initializer.ConstantInitializer(value=1.0),
            bias_attr=fluid.initializer.ConstantInitializer(value=1e-4),
            act="relu")
        conv2 = fluid.layers.conv2d_transpose(
            input=conv_bn1,
            num_filters=self.inner_channels // 4,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=self._get_bias_attr(0.0004, conv_bn1.shape[1], "conv2"),
            act=None)
        conv_bn2 = fluid.layers.batch_norm(
            input=conv2,
            param_attr=fluid.initializer.ConstantInitializer(value=1.0),
            bias_attr=fluid.initializer.ConstantInitializer(value=1e-4),
            act="relu")
        conv3 = fluid.layers.conv2d_transpose(
            input=conv_bn2,
            num_filters=1,
            filter_size=2,
            stride=2,
            param_attr=fluid.initializer.MSRAInitializer(uniform=False),
            bias_attr=self._get_bias_attr(0.0004, conv_bn2.shape[1], "conv3"),
            act=None)
        out = fluid.layers.sigmoid(conv3)
        return out

    def _get_bias_attr(self, l2_decay, k, name, gradient_clip=None):
        regularizer = fluid.regularizer.L2Decay(l2_decay)
        stdv = 1.0 / math.sqrt(k * 1.0)
        initializer = fluid.initializer.Uniform(-stdv, stdv)
        bias_attr = fluid.ParamAttr(
            regularizer=regularizer,
            initializer=initializer,
            name=name + "_b_attr")
        return bias_attr

    def step_function(self, x, y):
        return fluid.layers.reciprocal(1 + fluid.layers.exp(-self.k * (x - y)))

    def __call__(self, conv_features, mode="train"):
        """
        Fuse different levels of feature map from backbone in the FPN manner.
        Args:
            conv_features(list): feature maps from backbone
            mode(str): runtime mode, can be "train", "eval" or "test"
        Return: predicts
        """
        c2, c3, c4, c5 = conv_features
        param_attr = fluid.initializer.MSRAInitializer(uniform=False)
        in5 = fluid.layers.conv2d(
            input=c5,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)
        in4 = fluid.layers.conv2d(
            input=c4,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)
        in3 = fluid.layers.conv2d(
            input=c3,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)
        in2 = fluid.layers.conv2d(
            input=c2,
            num_filters=self.inner_channels,
            filter_size=1,
            param_attr=param_attr,
            bias_attr=False)

        out4 = fluid.layers.elementwise_add(
            x=fluid.layers.resize_nearest(
                input=in5, scale=2), y=in4)  # 1/16
        out3 = fluid.layers.elementwise_add(
            x=fluid.layers.resize_nearest(
                input=out4, scale=2), y=in3)  # 1/8
        out2 = fluid.layers.elementwise_add(
            x=fluid.layers.resize_nearest(
                input=out3, scale=2), y=in2)  # 1/4

        p5 = fluid.layers.conv2d(
            input=in5,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)
        p5 = fluid.layers.resize_nearest(input=p5, scale=8)
        p4 = fluid.layers.conv2d(
            input=out4,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)
        p4 = fluid.layers.resize_nearest(input=p4, scale=4)
        p3 = fluid.layers.conv2d(
            input=out3,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)
        p3 = fluid.layers.resize_nearest(input=p3, scale=2)
        p2 = fluid.layers.conv2d(
            input=out2,
            num_filters=self.inner_channels // 4,
            filter_size=3,
            padding=1,
            param_attr=param_attr,
            bias_attr=False)

        fuse = fluid.layers.concat(input=[p5, p4, p3, p2], axis=1)
        shrink_maps = self.binarize(fuse)
        if mode != "train":
            return {"maps": shrink_maps}



class MobileNetV3():
    def __init__(self, scale=0.5,model_name="large",disable_se=False):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        self.scale = scale
        self.inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 2],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                [5, 672, 160, True, 'hard_swish', 2],
                [5, 960, 160, True, 'hard_swish', 1],
                [5, 960, 160, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', 2],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                [5, 288, 96, True, 'hard_swish', 2],
                [5, 576, 96, True, 'hard_swish', 1],
                [5, 576, 96, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert self.scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, self.scale)

        self.disable_se = disable_se

    def __call__(self, input):
        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        cls_ch_squeeze = self.cls_ch_squeeze
        cls_ch_expand = self.cls_ch_expand
        # conv1
        conv = self.conv_bn_layer(
            input,
            filter_size=3,
            num_filters=self.make_divisible(inplanes * scale),
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv1')
        i = 0
        inplanes = self.make_divisible(inplanes * scale)
        outs = []
        for layer_cfg in cfg:
            if layer_cfg[5] == 2 and i > 2:
                outs.append(conv)
            conv = self.residual_unit(
                input=conv,
                num_in_filter=inplanes,
                num_mid_filter=self.make_divisible(scale * layer_cfg[1]),
                num_out_filter=self.make_divisible(scale * layer_cfg[2]),
                act=layer_cfg[4],
                stride=layer_cfg[5],
                filter_size=layer_cfg[0],
                use_se=layer_cfg[3],
                name='conv' + str(i + 2))
            inplanes = self.make_divisible(scale * layer_cfg[2])
            i += 1

        conv = self.conv_bn_layer(
            input=conv,
            filter_size=1,
            num_filters=self.make_divisible(scale * cls_ch_squeeze),
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv_last')
        outs.append(conv)
        return outs

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      if_act=True,
                      act=None,
                      name=None,
                      use_cudnn=True,
                      res_last_bn_init=False):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(
                name=bn_name + "_scale",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            bias_attr=ParamAttr(
                name=bn_name + "_offset",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            if act == 'relu':
                bn = fluid.layers.relu(bn)
            elif act == 'hard_swish':
                bn = fluid.layers.hard_swish(bn)
        return bn

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def se_block(self, input, num_out_filter, ratio=4, name=None):
        num_mid_filter = num_out_filter // ratio
        pool = fluid.layers.pool2d(
            input=input, pool_type='avg', global_pooling=True, use_cudnn=False)
        conv1 = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=num_mid_filter,
            act='relu',
            param_attr=ParamAttr(name=name + '_1_weights'),
            bias_attr=ParamAttr(name=name + '_1_offset'))
        conv2 = fluid.layers.conv2d(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            act='hard_sigmoid',
            param_attr=ParamAttr(name=name + '_2_weights'),
            bias_attr=ParamAttr(name=name + '_2_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        return scale

    def residual_unit(self,
                      input,
                      num_in_filter,
                      num_mid_filter,
                      num_out_filter,
                      stride,
                      filter_size,
                      act=None,
                      use_se=False,
                      name=None):

        conv0 = self.conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_mid_filter,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + '_expand')

        conv1 = self.conv_bn_layer(
            input=conv0,
            filter_size=filter_size,
            num_filters=num_mid_filter,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            if_act=True,
            act=act,
            num_groups=num_mid_filter,
            use_cudnn=False,
            name=name + '_depthwise')
        if use_se and not self.disable_se:
            conv1 = self.se_block(
                input=conv1, num_out_filter=num_mid_filter, name=name + '_se')

        conv2 = self.conv_bn_layer(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            stride=1,
            padding=0,
            if_act=False,
            name=name + '_linear',
            res_last_bn_init=True)
        if num_in_filter != num_out_filter or stride != 1:
            return conv2
        else:
            return fluid.layers.elementwise_add(x=input, y=conv2, act=None)


class DBModel(object):
    def __init__(self,image_shape):
        self.backbone = MobileNetV3()
        self.head = DBHead(image_shape)
        self.loss = DBLoss()
        self.image_shape = image_shape


    def create_feed(self):
        """
        create Dataloader feeds
        args:
            mode (str): 'train' for training  or else for evaluation
        return: (image, corresponding label, dataloader)
        """
        image_shape = deepcopy(self.image_shape)
        if image_shape[1] % 4 != 0 or image_shape[2] % 4 != 0:
            raise Exception("The size of the image must be divisible by 4, "
                            "received image shape is {}, please reset the "
                            "Global.image_shape in the yml file".format(
                                image_shape))

        image = fluid.layers.data(
            name='image', shape=image_shape, dtype='float32')
        image.stop_gradient = False
        labels = None
        loader = None
        return image, labels, loader

    def __call__(self, mode):
        """
        run forward of defined module
        args:
            mode (str): 'train' for training; 'export'  for inference,
                others for evaluation]
        """
        image, labels, loader = self.create_feed()
        conv_feas = self.backbone(image)
        predicts = self.head(conv_feas, mode)
        return loader, predicts


class DBTextDetector(object):
    def __init__(self,model_path,image_shape=[3, 640, 640]):
        self.model_path = model_path
        self.image_shape = image_shape
        self.min_size = 3
        self.dilation_kernel = np.array([[1, 1], [1, 1]])
        self.init_net()
        self.load_model()

        super(DBTextDetector, self).__init__()

    def init_net(self):
        place = fluid.CUDAPlace(0)
        self.exe = fluid.Executor(place)
        startup_prog = fluid.Program()
        self.eval_prog = fluid.Program()

        with fluid.program_guard(self.eval_prog, startup_prog):
            with fluid.unique_name.guard():
                _, eval_outputs = DBModel(self.image_shape)(mode="test")
                fetch_name_list = list(eval_outputs.keys())
                self.eval_fetch_list = [eval_outputs[v].name for v in fetch_name_list]

        self.eval_prog = self.eval_prog.clone(for_test=True)
        self.exe.run(startup_prog)

    def load_model(self):
        fluid.load(self.eval_prog, self.model_path, self.exe)

    def resize_image_type1(self, im):
        _,resize_h, resize_w = self.image_shape
        ori_h, ori_w = im.shape[:2]  # (h, w, c)
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        return im, (ratio_h, ratio_w)

    def normalize(self, im):
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im[:, :, 0] -= img_mean[0]
        im[:, :, 1] -= img_mean[1]
        im[:, :, 2] -= img_mean[2]
        im[:, :, 0] /= img_std[0]
        im[:, :, 1] /= img_std[1]
        im[:, :, 2] /= img_std[2]
        channel_swap = (2, 0, 1)
        im = im.transpose(channel_swap)
        return im

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded


    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours, ), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def postprocess(self,outs_dict,ratio_list):
        pred = outs_dict['maps']

        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = pred.shape[-2:]

            mask = cv2.dilate(np.array(segmentation[batch_index]).astype(np.uint8), self.dilation_kernel)
            tmp_boxes, tmp_scores = self.boxes_from_bitmap(pred[batch_index], mask, width, height)

            boxes = []
            for k in range(len(tmp_boxes)):
                if tmp_scores[k] > self.box_thresh:
                    boxes.append(tmp_boxes[k])
            if len(boxes) > 0:
                boxes = np.array(boxes)

                ratio_h, ratio_w = ratio_list[batch_index]
                boxes[:, :, 0] = boxes[:, :, 0] / ratio_w
                boxes[:, :, 1] = boxes[:, :, 1] / ratio_h

            boxes_batch.append(boxes)
        return boxes_batch
    def predict(self,image,thresh=0.3,box_thresh=0.3,max_candidates=1000,unclip_ratio=1.5):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        image, ratio_list = self.resize_image_type1(image)
        image = self.normalize(image)
        image = image[np.newaxis, :]
        img_list = np.concatenate([image], axis=0)
        outs = self.exe.run(self.eval_prog, \
                       feed={'image': img_list}, \
                       fetch_list=self.eval_fetch_list)
        dic = {'maps': outs[0]}
        dt_boxes = self.postprocess(dic,[ratio_list])
        return dt_boxes[0]
if __name__ == '__main__':
    import  cv2
    from jade import draw_ocr,GetAllImagesPath
    dbTextDetector=DBTextDetector(r"G:\SVN\软件\箱号识别服务\箱门检测+箱号识别服务V2.4.2\Windows\models\TextDetModels\2021-01-24")
    image_path_list = GetAllImagesPath(r"F:\数据集\字符检测数据集\苏州电子围网车牌关键点数据集\2021-08-31\image")
    for image_path in image_path_list:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                             -1)
        dt_boxes = dbTextDetector.predict(image)
        image = draw_ocr(image, dt_boxes, len(dt_boxes) * ["car_plate"], len(dt_boxes) * [1], draw_txt=False)
        cv2.namedWindow("result", 0)
        cv2.imshow("result", image)
        cv2.waitKey(0)


