# CIFAR-10 三层神经网络分类器
从零开始实现的三层神经网络，用于CIFAR-10图像分类任务。
cifar10_nn/
│── cifar10_nn/
│   │── __init__.py
│   │── model.py        # 神经网络模型实现
│   │── utils.py        # 数据处理工具
│   │── train.py        # 训练逻辑
│   │── config.py       # 配置文件
│── scripts/
│   │── train.py        # 训练脚本
│   │── test.py         # 测试脚本
|   |── hparam_search.py
│── README.md
│── setup.py

output/
├── best_model.npz          # 模型权重
├── training_results.json    # 训练记录和配置
├── training_curves.png      # 训练曲线图
└── test_results.json        # 测试后生成的结果文件

## 安装
```bash
pip install -e .
```

## 下载数据:

从https://www.cs.toronto.edu/~kriz/cifar.html下载CIFAR-10 Python版本
解压到data/cifar-10-batches-py目录

## 参数搜索:
python scripts/hparam_search.py

## 训练模型:
python scripts/train.py --hidden1 512 --hidden2 256 --lr 0.01 --reg 0.1（需调整为学习的最好参数）

## 测试模型:
python scripts/test.py --model_dir output
