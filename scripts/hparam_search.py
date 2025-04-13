import itertools
import json
import os
import numpy as np
from cifar10_nn.model import ThreeLayerNN
from cifar10_nn.train import Trainer
from cifar10_nn.utils import load_cifar10, preprocess_data

def run_experiment(h1, h2, lr, reg, X_train, y_train, X_val, y_val):
    model = ThreeLayerNN(3072, h1, h2, 10, reg_lambda=reg)
    trainer = Trainer(model, X_train, y_train, X_val, y_val, f'hparam_search/h1_{h1}_h2_{h2}_lr_{lr}_reg_{reg}')
    
    train_loss, val_loss, val_acc = trainer.train(
        epochs=10,  # 快速验证
        batch_size=64,
        learning_rate=lr,
        verbose=False
    )
    
    return {
        'h1': h1,
        'h2': h2,
        'lr': lr,
        'reg': reg,
        'final_val_acc': val_acc[-1],
        'final_val_loss': val_loss[-1]
    }

def main():
    # 定义搜索空间
    h1_list = [256, 512]
    h2_list = [128, 256]
    lr_list = [0.1, 0.05]  #学习率
    reg_list = [0.005, 0.01] #正则化强度
    
    # 加载数据
    X_train, y_train, X_val, y_val, _, _ = load_cifar10('data/cifar-10-batches-py')
    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)
    
    results = []
    
    for h1, h2, lr, reg in itertools.product(h1_list, h2_list, lr_list, reg_list):
        print(f"\nTesting h1={h1}, h2={h2}, lr={lr}, reg={reg}")
        result = run_experiment(h1, h2, lr, reg, X_train, y_train, X_val, y_val)
        results.append(result)
        
        # 保存中间结果
        os.makedirs('hparam_search', exist_ok=True)
        with open('hparam_search/results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # 找到最佳参数
    best = max(results, key=lambda x: x['final_val_acc'])
    print("\nBest Hyperparameters:")
    print(f"Hidden1: {best['h1']}, Hidden2: {best['h2']}")
    print(f"Learning Rate: {best['lr']}, Reg: {best['reg']}")
    print(f"Validation Accuracy: {best['final_val_acc']:.2%}")

if __name__ == '__main__':
    main()