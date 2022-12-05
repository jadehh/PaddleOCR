#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : test.py
# @Author   : jade
# @Date     : 2022/10/8 16:56
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from paddlelite.lite import *
import numpy as np
from PIL import Image
from rknn.api import RKNN

def test_paddle_Lite():


    # (1) 设置配置信息
    config = MobileConfig()
    config.set_model_from_file("models/container_rec_lite_opt/2021-12-07.nb")

    # (2) 创建预测器
    predictor = create_paddle_predictor(config)

    # (3) 从图片读入数据

    image_data = np.load("img.npy").astype("float32")

    # (4) 设置输入数据
    input_tensor = predictor.get_input(0)
    input_tensor.from_numpy(image_data)

    # (5) 执行预测
    predictor.run()

    # (6) 得到输出数据
    output_tensor = predictor.get_output(0)
    print(output_tensor.shape())
    print(output_tensor.numpy())

def test_onnx_to_rknn():


    rknn = RKNN(verbose=True)

    rknn.config(batch_size=1,optimization_level=3)

    ret = rknn.load_onnx("models/en_number_mobile_v2.0_rec_infer/model.onnx")
    print(ret)

    rknn.build(do_quantization=True, dataset="PATH_TO_YOUR_DATA.TXT", pre_compile=True)

    export_rknn_model_path = "models/container_rec_lite_onnx/2022-10-09/model" + "rknn"

    rknn.export_rknn(export_path=export_rknn_model_path)

    rknn.release()









if __name__ == '__main__':
    test_onnx_to_rknn()
