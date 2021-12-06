# PaddleOCR 

## 环境安装

```bash
pip install paddlepaddle_gpu-1.8.5.post107
pip install shapely
pip install imgaug
pip install pyclipper
```

## 训练检测模型

### 从头训练车牌关键点检测模型

```bash
python tools/train.py -c configs/det/car_plate/det_mv3_db.yml
```

### 恢复训练车牌关键点检测模型

```bash
python tools/train.py -c configs/det/car_plate/det_mv3_db.yml -o Global.checkpoints=./your/trained/model
```

### 从头训练箱号关键点检测点模型

```bash
python tools/train.py -c configs/det/det_mv3_container_db_v1.1.yml
```


## 测试文件夹下所有图片

```bash
python tools/infer_det.py -c configs/det/car_plate/det_mv3_db.yml -o TestReader.infer_img="/mnt/f/数据集/字符检测数据集/苏州电子围网车牌关键点数据集/2021-08-31/image" Global.checkpoints="/mnt/e/Models/Paddle1.8/字符检测模型/车牌检测模型/DBDet/best_accuracy"
```



## 车牌检测模型导出


```bash
python tools/export_model.py -c configs/det/car_plate/det_mv3_db.yml -o Global.checkpoints=/mnt/e/Models/Paddle1.8/字符检测模型/车牌检测模型/DBDet/best_model Global.save_inference_dir=/mnt/e/Models/Paddle1.8/字符检测模型/车牌检测模型/DBDet/2021-08-31
```


## 车牌检测模型推理
```bash
python tools/infer/predict_det.py --image_dir="/mnt/f/数据集/字符检测数据集/苏州电子围网车牌关键点数据集/2021-08-31/image/" --det_model_dir="/mnt/e/Models/Paddle1.8/字符检测模型/车牌检测模型/DBDet/2021-08-31"
```