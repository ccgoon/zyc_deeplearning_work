import os
class Config:
    # 模型配置
    INPUT_SIZE = 3072  # 32x32x3
    HIDDEN1 = 512
    HIDDEN2 = 256
    OUTPUT_SIZE = 10
    ACTIVATION = 'relu'
    REG_LAMBDA = 0.1
    
    # 训练配置
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    LR_DECAY = 0.95
    
    # 数据路径
    DATA_DIR = 'data/cifar-10-batches-py'
    OUTPUT_DIR = 'output_512+256+0.1+0.01+0.95'