# PaddleOCR 

## 环境安装

```bash
pip install paddlepaddle_gpu
pip install shapely
pip install imgaug
pip install pyclipper
```

## 训练检测模型

### 从头训练箱号关键点检测模型

```bash
python tools/train.py -c configs/det/det_container_mv3_db.yml
```

## 从头训练箱号识别模型
```bash
python tools/train.py -c configs/rec/container/rec_container_ocrh_mv3_tps_bilstm_ctc.yml
```

## 恢复训练模型
### 恢复训练箱号关键点检测模型

```bash
python tools/train.py -c configs/rec/container/rec_container_ocrh_mv3_tps_bilstm_ctc.yml -o Global.checkpoints=
```

### 恢复训练箱号识别模型
```bash
python tools/train.py -c configs/rec/container/rec_container_ocrh_mv3_tps_bilstm_ctc.yml -o Global.pretrained_model=output/rec/mv3_tps_bilstm_ctc/latest
```

## 模型导出

### 箱号关键点检测模型导出

```bash
python tools/export_model.py -c configs/det/det_container_mv3_db.yml -o Global.pretrained_model="E:\Models\Paddle2.3\db箱号字符检测模型\2021-12-07-train/latest" Global.save_inference_dir="models/det_db_inference/2021-12-07"
```

### 箱号识别模型导出

```bash
_python tools/export_model.py -c configs/rec/container/rec_container_ocrh_mv3_tps_bilstm_ctc.yml -o Global.pretrained_model="output/rec/mv3_tps_bilstm_ctc/latest" Global.save_inference_dir="models/rec_inference/2021-12-07"_ 
```



## 预测


### 箱号识别模型预测

```bash
python tools/infer_rec.py -c configs/rec/container/rec_container_ocrh_mv3_tps_bilstm_ctc.yml -o Global.pretrained_model="output/rec/mv3_tps_bilstm_ctc/best_accuracy" Global.load_static_weights=false Global.infer_img=E:\Data\OCR\箱号识别数据集\OCRH\2020-03-22\train\A_160326111435_0_f04907a6-5722-11ec-bcc2-309c23add11a.jpg
python tools/infer/predict_rec.py --image_dir="E:\Data\OCR\箱号识别数据集\OCRH\2020-03-22\train\A_160326111435_0_f04907a6-5722-11ec-bcc2-309c23add11a.jpg" --rec_model_dir="models/rec_inference/2021-12-07" --rec_image_shape="3, 32, 400" --rec_char_type="ch" --rec_char_dict_path="ppocr/utils/containert.txt"
```



## 文本检测、方向分类和文字识别串联推理
以超轻量中文OCR模型推理为例，在执行预测时，需要通过参数image_dir指定单张图像或者图像集合的路径、参数det_model_dir,cls_model_dir和rec_model_dir分别指定检测，方向分类和识别的inference模型路径。参数use_angle_cls用于控制是否启用方向分类模型。use_mp表示是否使用多进程。total_process_num表示在使用多进程时的进程数。可视化识别结果默认保存到 ./inference_results 文件夹里面。

```bash
python tools/infer/predict_system.py --image_dir="C:\Users\Administrator\Desktop\test_images/" --det_model_dir="E:\Models\Paddle2.3\ch_PP-OCRv2_det_infer/"  --rec_model_dir="E:\Models\Paddle2.3\ch_PP-OCRv2_rec_infer/" --use_angle_cls=false
```
