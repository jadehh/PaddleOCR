# PaddleOCR 

## 环境安装

```bash
pip install paddlepaddle_gpu-1.8.5.post107
pip install shapely
pip install imgaug
pip install pyclipper
```

## 训练车牌检测模型

### 从头训练

```bash
python tools/train.py -c configs/det/car_plate/det_mv3_db.yml
```