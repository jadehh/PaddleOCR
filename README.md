# Paddle OCR 2.1


## 训练

```bash
python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0'  tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml
```




## 恢复训练

```bash
python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0'  tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml  -o Global.checkpoints=/home/samples/sda2/Models/Paddle2.1/箱号箱型字符检测/last
```