# PaddleOCR

## 环境安装

```bash
pip install paddle1.8
pip install imgaug
```

## 训练车牌检测模型

### 从头训练

```bash
python tools/train.py -c configs/det/car_plate/det_mv3_db.yml
```