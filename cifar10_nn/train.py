import numpy as np
import os
import json
from tqdm import tqdm
from .utils import visualize_weights

class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, output_dir='output'):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.best_metrics = {
            'epoch': -1,
            'val_acc': 0,
            'val_loss': float('inf'),
            'train_loss': float('inf')
        }
    
    def train(self, epochs=100, batch_size=64, learning_rate=0.01, 
              lr_decay=0.95, verbose=True):
        train_loss_history = []
        val_loss_history = []
        val_acc_history = []
        
        n_train = (self.X_train.shape[0] // batch_size) * batch_size
        X_train = self.X_train[:n_train]
        y_train = self.y_train[:n_train]
        
        for epoch in tqdm(range(epochs), desc="Training"):
            learning_rate *= lr_decay
            
            indices = np.random.permutation(n_train)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            
            for i in range(0, n_train, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                probs, cache = self.model.forward(X_batch)
                
                correct_logprobs = -np.log(probs[range(batch_size), y_batch])
                data_loss = np.sum(correct_logprobs)/batch_size
                reg_loss = 0.5 * self.model.reg_lambda * (
                    np.sum(self.model.params['W1']**2) + 
                    np.sum(self.model.params['W2']**2) + 
                    np.sum(self.model.params['W3']**2))
                loss = data_loss + reg_loss
                epoch_loss += loss
                
                grads = self.model.backward(X_batch, y_batch, cache)
                
                self.model.params['W1'] -= learning_rate * grads['dW1']
                self.model.params['b1'] -= learning_rate * grads['db1']
                self.model.params['W2'] -= learning_rate * grads['dW2']
                self.model.params['b2'] -= learning_rate * grads['db2']
                self.model.params['W3'] -= learning_rate * grads['dW3']
                self.model.params['b3'] -= learning_rate * grads['db3']
            
            val_probs, _ = self.model.forward(self.X_val)
            val_loss = -np.log(val_probs[range(self.X_val.shape[0]), self.y_val]).mean()
            val_preds = np.argmax(val_probs, axis=1)
            val_acc = np.mean(val_preds == self.y_val)
            
            train_loss_history.append(epoch_loss/(n_train//batch_size))
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            
            if val_acc > self.best_metrics['val_acc']:
                self.best_metrics = {
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_loss': train_loss_history[-1]
                }
                self.model.save(os.path.join(self.output_dir, 'best_model.npz'))
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss_history[-1]:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        self._save_results(train_loss_history, val_loss_history, val_acc_history)
        return train_loss_history, val_loss_history, val_acc_history
    
    def _save_results(self, train_loss, val_loss, val_acc):
        results = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_metrics': self.best_metrics,
            'config': {
                'input_size': self.model.params['W1'].shape[0],
                'hidden1': self.model.params['W1'].shape[1],
                'hidden2': self.model.params['W2'].shape[1],
                'output_size': self.model.params['W3'].shape[1],
                'activation': self.model.activation,
                'reg_lambda': self.model.reg_lambda
            }
        }
        
        with open(os.path.join(self.output_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # 新增参数可视化
        visualize_weights(
            self.model.params,
            save_path=os.path.join(self.output_dir, 'weight_visualization.png')
        )
        