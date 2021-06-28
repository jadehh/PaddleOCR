# Paddle OCR 2.1


## 训练

```bash
python train.py -m paddle.distributed.launch --log_dir=./debug/ --gpus '0'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml
```