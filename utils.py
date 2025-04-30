import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_cifar10(data_dir):
    X_train = []
    y_train = []
    
    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            X_train.append(data[b'data'])
            y_train.append(data[b'labels'])
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        X_test = data[b'data']
        y_test = np.array(data[b'labels'])
    
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-7)

def save_training_results(output_dir, train_loss, val_loss, val_acc, config):
    results = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': vars(config) if hasattr(config, '__dict__') else config
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

def visualize_weights(params, save_path=None):
    """
    可视化网络权重模式
    :param params: 模型参数字典 {'W1':..., 'b1':..., ...}
    :param save_path: 图片保存路径
    """
    plt.figure(figsize=(15, 10))
    
    # 第一层权重可视化 (输入→隐藏层1)
    W1 = params['W1']
    plt.subplot(2, 2, 1)
    plt.hist(W1.flatten(), bins=100)
    plt.title('Layer1 Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    # 第一层权重模式可视化 (前64个神经元)
    plt.subplot(2, 2, 2)
    w_min = np.min(W1)
    w_max = np.max(W1)
    # 将权重归一化并转为图像格式
    weights_rescaled = (W1[:, :64] - w_min) / (w_max - w_min)
    plt.imshow(weights_rescaled.T.reshape(-1, 32, 32, 3).transpose(0,2,1,3).reshape(8*32, 8*32, 3))
    plt.title('Layer1 Weight Patterns (First 64 neurons)')
    plt.colorbar()
    
    # 第二层权重分布
    W2 = params['W2']
    plt.subplot(2, 2, 3)
    plt.hist(W2.flatten(), bins=100)
    plt.title('Layer2 Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    # 层间权重相关性
    plt.subplot(2, 2, 4)
    plt.scatter(W1.flatten()[::1000], W2.flatten()[::1000], alpha=0.1)
    plt.title('Cross-layer Weight Correlation')
    plt.xlabel('Layer1 Weights')
    plt.ylabel('Layer2 Weights')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()