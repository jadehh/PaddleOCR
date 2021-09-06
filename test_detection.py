#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : test_detection.py
# @Author   : jade
# @Date     : 2021/9/1 9:29
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
from src.DBTextDetector import TextDetector
from jade import *
def get_image_path(root_path):
    image_path_list = []
    with open(os.path.join(root_path,"test_icdar2015_label.txt"),"rb") as f:
        content_list = f.readlines()
        for content in content_list:
            content = str(content,encoding="utf-8")
            image_path_list.append(os.path.join(root_path,content.split("\t")[0]))
    return image_path_list


if __name__ == '__main__':
    text_detector = TextDetector("models/2021-08-31", 1000)
    image_path_list = get_image_path("/mnt/f/数据集/字符检测数据集/苏州电子围网车牌关键点数据集")
    for image_path in image_path_list:
        image =  cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        dt_boxes = text_detector.predict(image)
        image = draw_ocr(image,dt_boxes,len(dt_boxes)*["car_Plate"],len(dt_boxes)*[1],draw_txt=False)
        print(dt_boxes)
        cv2.namedWindow("image",0)
        cv2.imshow('image', image)
        cv2.waitKey(0)
