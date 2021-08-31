# Paddle OCR

## 训练

### 训练水平箱号识别模型

```bash
python train.py -c configs/rec/container/rec_contanum_h_train_ctc.yml
python train.py -c configs/rec/container/rec_contanum_h_mv3_tps_bilstm_ctc.yml

```

### 训练垂直箱号识别模型

```bash
python train.py -c configs/rec/container/rec_contanum_v_train_ctc.yml
```

## 恢复训练

### 恢复训练水平箱号识别模型

```bash
python train.py -c configs/rec/container/rec_contanum_h_train_ctc.yml -o Global.checkpoints=E:\Models\Paddle2.1\箱号识别模型\水平箱号识别模型\CRNN-CTC\2021-08-06\latest
```

## 模型导出


### 导出水平箱号识别模型

```bash
python export_model.py -c configs/rec/container/rec_contanum_h_train_ctc.yml -o Global.checkpoints=/mnt/e/Models/Paddle2.1/箱号识别模型/水平箱号识别模型/CRNN-CTC/2021-08-06/latest Global.save_inference_dir=/mnt/e/Models/Paddle2.1/箱号识别模型/水平箱号识别模型/CRNN-CTC/2021-08-06/2021-08-06
```