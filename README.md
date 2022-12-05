# PaddleOCR

## 训练

### 训练箱号字符检测模型
```bash
python tools/train.py -c configs/det/container/container_PP-OCRv2_det_distill.yml --eval
python tools/train.py -c configs/det/container/container_det_mv3_db.yml
python tools/train.py -c configs/det/container/container_det_mv3_db_v2.0_512_384.yml

```

### 训练水平箱号识别模型
```bash
python tools/train.py -c configs/rec/container_number/container_number_h/container_number_rec_SVTR_H.yml 
python tools/train.py -c configs/rec/container_number/container_number_h/container_number_rec_CRNN_H.yml
python tools/train.py -c configs/rec/container_number/container_number_h/container_number_rec_SVTR-Tiny_H.yml
python tools/train.py -c configs/rec/container_number/container_number_h/container_number_PP-OCRv3_rec_H.yml 
python tools/train.py -c configs/rec/container_number/container_number_h/container_number_PP-OCRv3_rec_distillation-H.yml
python tools/train.py -c configs/rec/container_number/container_number_h/container_rec_lite_train_H.yml
python tools/train.py -c configs/rec/container_number/container_number_h/container_rec_lite_train_v2.0_H.yml
python tools/train.py -c configs/rec/container_number/container_number_h/container_number_PP-OCRv2_rec.yml

```

### 训练垂直箱号识别模型

```bash
python tools/train.py -c configs/rec/container_number/container_number_v/container_rec_lite_train_V.yml
```

## 恢复训练

### 恢复训练水平箱号识别模型

```bash
python tools/train.py -c configs/rec/container_number/container_number_h/container_number_rec_SVTR-Tiny_H.yml -o Global.pretrained_model=output/rec/container_number_rec_SVTR-Tiny_H/best_accuracy
python tools/train.py -c configs/rec/container_number/container_number_h/container_rec_lite_train_H.yml -o Global.pretrained_model=output/rec/container_rec_lite/latest
python tools/train.py -c configs/rec/container_number/container_number_h/container_number_PP-OCRv2_rec.yml -o Global.pretrained_model=output/rec/container_number_PP-OCRv2_rec/latest

```

### 恢复训练箱号字符检测模型

```bash 

python tools/train.py -c configs/det/container/container_det_mv3_db.yml -o Global.pretrained_model=output/det/container_det_mv3_db/latest
python tools/train.py -c configs/det/container/container_det_mv3_db_v2.0_512_384.yml -o Global.pretrained_model=output/det/container/container_det_mv3_db_v2.0_512_384/latest

```

## 模型导出

### 导出水平箱号识别模型

```bash
python tools/export_model.py -c configs/rec/container_number/container_number_h/container_number_rec_SVTR-Tiny_H.yml -o Global.pretrained_model=output/rec/container_number_rec_SVTR-Tiny_H/best_accuracy  Global.save_inference_dir=models/container_number_rec_SVTR-Tiny_H/ 
python tools/export_model.py -c configs/rec/container_number/container_number_h/container_rec_lite_train_H.yml -o Global.pretrained_model=output/rec/container_rec_lite/best_accuracy  Global.save_inference_dir=models/container_rec_lite/2022-10-09/
python tools/export_model.py -c configs/rec/container_number/container_number_h/container_number_rec_SVTR-Tiny_H.yml -o Global.pretrained_model=output/rec/container_number_rec_SVTR-Tiny_H/best_accuracy  Global.save_inference_dir=models/container_number_rec_SVTR-Tiny_H/2022-10-09/
python tools/export_model.py -c configs/rec/container_number/container_number_h/container_rec_lite_train_v2.0_H.yml -o Global.pretrained_model=output/rec/container_rec_lite_train_v2.0/latest  Global.save_inference_dir=models/container_rec_lite_v2.0/2022-10-10
python tools/export_model.py -c configs/rec/container_number/container_number_h/container_number_PP-OCRv2_rec.yml -o Global.pretrained_model=output/rec/container_number_PP-OCRv2_rec/best_accuracy  Global.save_inference_dir=models/container_number_PP-OCRv2_rec/2022-11-08


```
### 导出垂直箱号识别模型
```bash
python tools/export_model.py -c configs/rec/container_number/container_number_v/container_rec_lite_train_V.yml  -o Global.pretrained_model=output/rec/container_number_v/container_rec_lite_train_V/best_accuracy Global.save_inference_dir=models/container_number_v/container_rec_lite_train_V/2022-10-09
``` 


### 导出箱号字符检测模型
```bash
python tools/export_model.ppython tools/export_model.py -c configs/det/container/container_det_mv3_db.yml  -o Global.pretrained_model=output/det/container_det_mv3_db/best_accuracy Global.save_inference_dir=models/container_det_mv3_db/2022-10-09/
python tools/export_model.py -c configs/det/container/container_det_mv3_db_v2.0_512_384.yml  -o Global.pretrained_model=output/det/container/container_det_mv3_db_v2.0_512_384/best_accuracy Global.save_inference_dir=models/det/container/container_det_mv3_db_v2.0_512_384/2022-11-18/

```

## 测试

### 测试水平箱号识别模型

```bash
python tools/infer_rec.py -c configs/rec/container_number/container_number_h/container_number_rec_SVTR-Tiny_H.yml -o Global.pretrained_model=output/rec/container_number_rec_SVTR-Tiny_H/best_accuracy Global.infer_img=E:\Data\OCR\container_number\OCRH\2022-09-07\train
python tools/infer/predict_rec.py --rec_image_shape=3,80,400 --image_dir=E:\Data\OCR\container_number\OCRH\2022-09-07\train --rec_model_dir=inference/container_number_rec_SVTR-Tiny_H --rec_char_dict_path=chardicts/containernumber.txt 
python predict_rec.py  --rec_image_shape=3,32,320 --dataset_dir=E:\Data\OCR\container_number\OCRH\  --rec_model_dir=models/container_rec_lite/2022-10-09 --rec_char_dict_path=chardicts/containernumber.txt
python predict_rec.py  --rec_image_shape=3,48,800 --dataset_dir=E:\Data\OCR\container_number\OCRH\  --rec_model_dir=models/container_number_rec_SVTR-Tiny_H/2022-10-09 --rec_char_dict_path=chardicts/containernumber.txt
python predict_rec.py  --rec_image_shape=3,32,320 --dataset_dir=E:\Data\OCR\container_number\OCRH\  --rec_model_dir=models/container_rec_lite/2022-10-09 --rec_char_dict_path=chardicts/containernumber.txt

```

### 测试垂直箱号识别模型

```bash
python predict_rec.py  --rec_image_shape=3,32,320 --dataset_dir=E:\Data\OCR\container_number\OCRV\  --rec_model_dir=models/container_number_v/container_rec_lite_train_V/2022-10-09 --rec_char_dict_path=chardicts/containernumber.txt
```



