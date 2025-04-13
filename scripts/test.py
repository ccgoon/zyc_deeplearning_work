import os
import argparse
import json
import numpy as np
from cifar10_nn.model import ThreeLayerNN
from cifar10_nn.utils import load_cifar10, preprocess_data

def load_model_config(model_dir):
    """从指定目录加载模型配置"""
    config_path = os.path.join(model_dir, 'training_results.json')
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data['config']

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    probs, _ = model.forward(X_test)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == y_test)
    loss = -np.log(probs[np.arange(len(y_test)), y_test]).mean()
    return accuracy, loss, preds

def save_test_results(model_dir, accuracy, loss, preds):
    """保存测试结果到文件"""
    results = {
        'test_accuracy': float(accuracy),
        'test_loss': float(loss),
        'predictions': preds.tolist()
    }
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='Test a 3-layer NN on CIFAR-10')
    parser.add_argument('--model_dir', type=str, default='output_512+256+0.1+0.01+0.95',
                      help='Directory containing trained model files')
    parser.add_argument('--data_dir', type=str, default='data/cifar-10-batches-py',
                      help='Directory containing CIFAR-10 dataset')
    args = parser.parse_args()

    # 验证模型目录是否存在
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    # 加载模型配置
    config = load_model_config(args.model_dir)
    
    # 初始化模型结构
    model = ThreeLayerNN(
        input_size=config['input_size'],
        hidden1=config['hidden1'],
        hidden2=config['hidden2'],
        output_size=config['output_size'],
        activation=config['activation'],
        reg_lambda=config['reg_lambda']
    )
    
    # 加载模型权重
    model_path = os.path.join(args.model_dir, 'best_model.npz')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    model.load(model_path)

    # 加载并预处理测试数据
    _, _, _, _, X_test, y_test = load_cifar10(args.data_dir)
    X_test = preprocess_data(X_test)

    # 评估模型
    accuracy, loss, preds = evaluate_model(model, X_test, y_test)

    # 保存结果
    save_test_results(args.model_dir, accuracy, loss, preds)

    # 打印结果
    print(f"\nTest Results (saved to {args.model_dir}/test_results.json):")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Loss: {loss:.4f}")

if __name__ == '__main__':
    main()