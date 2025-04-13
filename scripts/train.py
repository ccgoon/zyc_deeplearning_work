import os
import argparse
import matplotlib.pyplot as plt
from cifar10_nn.model import ThreeLayerNN
from cifar10_nn.utils import load_cifar10, preprocess_data
from cifar10_nn.train import Trainer

def main():
    parser = argparse.ArgumentParser(description='Train a 3-layer NN on CIFAR-10')
    parser.add_argument('--data_dir', type=str, default='data/cifar-10-batches-py',
                       help='Directory containing CIFAR-10 dataset')
    parser.add_argument('--output_dir', type=str, default='output_512+256+0.1+0.01+0.95',
                       help='Directory to save model and results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                       help='Learning rate decay per epoch')
    parser.add_argument('--hidden1', type=int, default=512,
                       help='Number of neurons in first hidden layer')
    parser.add_argument('--hidden2', type=int, default=256,
                       help='Number of neurons in second hidden layer')
    parser.add_argument('--reg', type=float, default=0.1,
                       help='L2 regularization strength')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'sigmoid', 'tanh'],
                       help='Activation function type')
    args = parser.parse_args()
    
    # 加载数据
    X_train, y_train, X_val, y_val, _, _ = load_cifar10(args.data_dir)
    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)
    
    # 初始化模型
    model = ThreeLayerNN(
        input_size=3072,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        output_size=10,
        activation=args.activation,
        reg_lambda=args.reg
    )
    
    # 训练模型
    trainer = Trainer(model, X_train, y_train, X_val, y_val, args.output_dir)
    train_loss, val_loss, val_acc = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_decay=args.lr_decay
    )
    
    # 绘制曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    plt.close()
    
    print("Training completed. Results saved to", args.output_dir)

if __name__ == '__main__':
    main()