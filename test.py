#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : test.py
# @Author   : jade
# @Date     : 2021/7/29 16:07
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from src.TextDetection import TextDetector,draw_text_det_res
import cv2
from src.detector import Detector
if __name__ == '__main__':
    model_dir = "/home/jade/sda2/Models/箱门检测模型/2021-06-28/2021-06-28/ppyolo_tiny_voc"
    from jade.jade_visualize import visualize
    from jade import *
    import imutils
    detector = Detector(
        model_dir,
        use_gpu=True,
        run_mode="fluid",
        batch_size=1,
        use_dynamic_shape=False,
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        trt_calib_mode=False,
        cpu_threads=10,
        enable_mkldnn=True)
    textDetector = TextDetector("/mnt/e/Models/Paddle2.1/箱号箱型字符检测/det_mv3_db/2021-07-29/2021-07-29")

    capture = cv2.VideoCapture("/mnt/f/视频数据集/箱号视频/常州/merge/left/2020-08-03-09-52-21-811.avi")
    while True:
        ret,frame = capture.read()
        frame = imutils.resize(frame, width=768)
        if ret is False:
            break
        results = detector.predict(frame,threshold=0.7,remove_class_type="UPEND")
        if results["labels"][0] != -1:
            dt_boxes, elapse = textDetector.predict(frame)
            frame = draw_text_det_res(dt_boxes, frame)
            frame = visualize(frame, results)
        cv2.namedWindow("result",0)
        cv2.imshow("result",frame)
        cv2.waitKey(0)

