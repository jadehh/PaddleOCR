# Paddle OCR 2.1


## 训练箱号箱型文本检测模型

```bash
python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0'  tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml
python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0'  tools/train.py -c configs/det/det_mv3_db.yml

```




## 恢复训练箱号箱型文本检测模型

```bash
python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0'  tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml  -o Global.checkpoints=/home/samples/sda2/Models/Paddle2.1/箱号箱型字符检测/last
python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0'  tools/train.py -c configs/det/det_mv3_db.yml  -o Global.checkpoints=/home/samples/sda2/Models/Paddle2.1/箱号箱型字符检测/det_mv3_db/2021-07-29/last

```


## 训练水平箱号识别模型

```bash
python3 -m paddle.distributed.launch --gpus '0'  tools/train.py -c configs/rec/rec_contanum_h_train_ctc.yml
python3 -m paddle.distributed.launch --gpus '0'  tools/train.py -c configs/rec/rec_contanum_h_train_att.yml

``


## 恢复训练水平箱号识别模型

```bash
python3 -m paddle.distributed.launch --gpus '0'  tools/train.py -c configs/rec/rec_contanum_h_train_ctc.yml  -o Global.checkpoints=/home/samples/sda2/Models/Paddle2.1/箱号识别模型/水平箱号识别模型/2021-07-28/latest
```




## 训练垂直箱号识别模型

```bash
python3 -m paddle.distributed.launch --gpus '0'  tools/train.py -c configs/rec/rec_contanum_v_train_ctc.yml

```

## 箱号箱型文本检测模型导出

```bash
python tools/export_model.py  -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml -o Global.checkpoints=/mnt/e/Models/Paddle2.1/箱号箱型字符检测/2021-07-22/latest Global.save_inference_dir=/mnt/e/Models/Paddle2.1/箱号箱型字符检测/2021-07-22/2021-07-22/

python tools/export_model.py  -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml -o Global.checkpoints=/mnt/e/Models/Paddle2.1/箱号箱型字符检测/ch_det_mv3_db_v2.0/2021-07-30/best_accuracy Global.save_inference_dir=/mnt/e/Models/Paddle2.1/箱号箱型字符检测/ch_det_mv3_db_v2.0/2021-07-30/2021-07-30/

python tools/export_model.py  -c configs/det/det_mv3_db.yml -o Global.checkpoints=/mnt/e/Models/Paddle2.1/箱号箱型字符检测/det_mv3_db/2021-07-29/best_accuracy Global.save_inference_dir=/mnt/e/Models/Paddle2.1/箱号箱型字符检测/det_mv3_db/2021-07-29/2021-07-29/

```

## 水平箱号识别模型导出

```bash
python tools/export_model.py  -c configs/rec/rec_contanum_h_train_ctc.yml -o Global.checkpoints=/mnt/e/Models/Paddle2.1/箱号识别模型/水平箱号识别模型/2021-07-28/best_accuracy Global.save_inference_dir=/mnt/e/Models/Paddle2.1/箱号识别模型/水平箱号识别模型/2021-07-28/2021-07-28/
```


## 箱号箱型文本检测模型评估

```
python tools/eval.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml -o Global.checkpoints=/mnt/e/Models/Paddle2.1/箱号箱型字符检测/2021-07-22/latest PostProcess.box_thresh=0.5 PostProcess.unclip_ratio=1.5
```


## 水平箱号识别模型评估

```bash
python tools/eval.py -c configs/rec/rec_contanum_h_train_ctc.yml -o Global.checkpoints=/mnt/e/Models/Paddle2.1/箱号识别模型/水平箱号识别模型/2021-07-28/best_accuracy
```
