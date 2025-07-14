import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class DynamicWeightNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_sizes, num_tasks, cache_size=20):
        super(DynamicWeightNetwork, self).__init__()
        
        self.num_tasks = num_tasks
        self.cache_size = cache_size
        self.hidden_sizes = hidden_sizes
        self.output_sizes = output_sizes  # Different output sizes for different tasks
        
        # Task Classifier (Decipher Layer) - More sophisticated for 10 tasks
        self.task_classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_tasks),
            nn.Softmax(dim=1)
        )
        
        # Weight Cache - larger cache for more complex task switching
        self.weight_cache = nn.ModuleDict()
        
        # Create weight cache for each task and layer
        layer_sizes = [input_size] + hidden_sizes
        for task_id in range(num_tasks):
            task_weights = nn.ModuleDict()
            
            # Hidden layers
            for layer_idx in range(len(layer_sizes) - 1):
                in_size = layer_sizes[layer_idx]
                out_size = layer_sizes[layer_idx + 1]
                
                layer_cache = nn.ModuleList([
                    nn.Linear(in_size, out_size) for _ in range(cache_size)
                ])
                task_weights[f'layer_{layer_idx}'] = layer_cache
            
            # Output layer - task-specific output sizes
            final_layer_cache = nn.ModuleList([
                nn.Linear(hidden_sizes[-1], output_sizes[task_id]) for _ in range(cache_size)
            ])
            task_weights[f'output_layer'] = final_layer_cache
            
            self.weight_cache[f'task_{task_id}'] = task_weights
        
        # Enhanced Weight Selector Networks
        self.weight_selectors = nn.ModuleDict()
        for task_id in range(num_tasks):
            task_selectors = nn.ModuleDict()
            
            # Selectors for hidden layers
            for layer_idx in range(len(layer_sizes) - 1):
                selector = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, cache_size),
                    nn.Softmax(dim=1)
                )
                task_selectors[f'layer_{layer_idx}'] = selector
            
            # Selector for output layer
            output_selector = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, cache_size),
                nn.Softmax(dim=1)
            )
            task_selectors[f'output_layer'] = output_selector
            
            self.weight_selectors[f'task_{task_id}'] = task_selectors
    
    def forward(self, x, task_label=None, use_hard_selection=False):
        batch_size = x.size(0)
        
        # Task classification (decipher layer)
        task_probs = self.task_classifier(x)
        
        if task_label is not None:
            # During training, use provided task labels
            task_distribution = F.one_hot(task_label, num_classes=self.num_tasks).float()
        else:
            # During inference, use predicted task distribution
            if use_hard_selection:
                task_ids = torch.argmax(task_probs, dim=1)
                task_distribution = F.one_hot(task_ids, num_classes=self.num_tasks).float()
            else:
                task_distribution = task_probs
        
        # Forward pass through dynamically selected weights
        current_input = x
        
        # Process hidden layers
        for layer_idx in range(len(self.hidden_sizes)):
            layer_output = torch.zeros(batch_size, self.hidden_sizes[layer_idx], device=x.device)
            
            for task_id in range(self.num_tasks):
                task_weight = task_distribution[:, task_id].unsqueeze(1)
                
                weight_selector = self.weight_selectors[f'task_{task_id}'][f'layer_{layer_idx}']
                weight_probs = weight_selector(x)
                
                cached_layers = self.weight_cache[f'task_{task_id}'][f'layer_{layer_idx}']
                
                task_layer_output = torch.zeros_like(layer_output)
                for cache_idx, cached_layer in enumerate(cached_layers):
                    cache_weight = weight_probs[:, cache_idx].unsqueeze(1)
                    task_layer_output += cache_weight * cached_layer(current_input)
                
                layer_output += task_weight * task_layer_output
            
            current_input = F.relu(layer_output)
        
        # Process output layer - handle different output sizes
        max_output_size = max(self.output_sizes)
        final_outputs = []
        
        for task_id in range(self.num_tasks):
            task_weight = task_distribution[:, task_id]
            
            weight_selector = self.weight_selectors[f'task_{task_id}'][f'output_layer']
            weight_probs = weight_selector(x)
            
            cached_layers = self.weight_cache[f'task_{task_id}'][f'output_layer']
            
            task_output = torch.zeros(batch_size, self.output_sizes[task_id], device=x.device)
            for cache_idx, cached_layer in enumerate(cached_layers):
                cache_weight = weight_probs[:, cache_idx].unsqueeze(1)
                task_output += cache_weight * cached_layer(current_input)
            
            # Pad smaller outputs to max size for consistent tensor operations
            if self.output_sizes[task_id] < max_output_size:
                padding = torch.zeros(batch_size, max_output_size - self.output_sizes[task_id], device=x.device)
                task_output = torch.cat([task_output, padding], dim=1)
            
            final_outputs.append(task_output)
        
        # Combine outputs based on task distribution
        combined_output = torch.zeros(batch_size, max_output_size, device=x.device)
        for task_id, task_output in enumerate(final_outputs):
            task_weight = task_distribution[:, task_id].unsqueeze(1)
            combined_output += task_weight * task_output
        
        return combined_output, task_probs
    
    def get_task_prediction(self, x):
        with torch.no_grad():
            task_probs = self.task_classifier(x)
            return torch.argmax(task_probs, dim=1)

def create_diverse_datasets(n_samples=1000, input_size=100):
    """Create 10 diverse datasets for different tasks"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    datasets = {}
    task_types = {}
    
    # Task 0: MNIST-like digit classification (10 classes)
    X0 = torch.randn(n_samples, input_size)
    y0 = torch.randint(0, 10, (n_samples,))
    datasets[0] = (X0, y0)
    task_types[0] = 'classification'
    
    # Task 1: Binary classification 
    X1, y1 = make_classification(n_samples=n_samples, n_features=input_size, n_classes=2, 
                                n_informative=20, n_redundant=10, random_state=42)
    datasets[1] = (torch.FloatTensor(X1), torch.LongTensor(y1))
    task_types[1] = 'classification'
    
    # Task 2: Multi-class classification (5 classes)
    X2, y2 = make_classification(n_samples=n_samples, n_features=input_size, n_classes=5, 
                                n_informative=30, n_clusters_per_class=2, random_state=42)
    datasets[2] = (torch.FloatTensor(X2), torch.LongTensor(y2))
    task_types[2] = 'classification'
    
    # Task 3: Regression - Single output
    X3, y3 = make_regression(n_samples=n_samples, n_features=input_size, n_targets=1, 
                            noise=0.1, random_state=42)
    datasets[3] = (torch.FloatTensor(X3), torch.FloatTensor(y3).unsqueeze(1))
    task_types[3] = 'regression'
    
    # Task 4: Multi-output regression (3 outputs)
    X4, y4 = make_regression(n_samples=n_samples, n_features=input_size, n_targets=3, 
                            noise=0.1, random_state=42)
    datasets[4] = (torch.FloatTensor(X4), torch.FloatTensor(y4))
    task_types[4] = 'regression'
    
    # Task 5: Clustering-based classification (8 classes)
    X5, y5 = make_blobs(n_samples=n_samples, centers=8, n_features=input_size, 
                       cluster_std=1.0, random_state=42)
    datasets[5] = (torch.FloatTensor(X5), torch.LongTensor(y5))
    task_types[5] = 'classification'
    
    # Task 6: Polynomial regression
    X6 = torch.randn(n_samples, input_size)
    y6 = torch.sum(X6[:, :5]**2, dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    datasets[6] = (X6, y6)
    task_types[6] = 'regression'
    
    # Task 7: XOR-like complex classification (2 classes)
    X7 = torch.randn(n_samples, input_size)
    y7 = ((X7[:, 0] * X7[:, 1]) > 0).long()
    datasets[7] = (X7, y7)
    task_types[7] = 'classification'
    
    # Task 8: Multi-class with many classes (15 classes)
    X8, y8 = make_classification(n_samples=n_samples, n_features=input_size, n_classes=15, 
                                n_informative=50, n_redundant=20, random_state=42)
    datasets[8] = (torch.FloatTensor(X8), torch.LongTensor(y8))
    task_types[8] = 'classification'
    
    # Task 9: Time series-like regression
    X9 = torch.randn(n_samples, input_size)
    y9 = torch.cumsum(X9[:, :10], dim=1)[:, -1:] + 0.1 * torch.randn(n_samples, 1)
    datasets[9] = (X9, y9)
    task_types[9] = 'regression'
    
    return datasets, task_types

def train_multi_task_network(model, train_loaders, task_types, output_sizes, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    task_criterion = nn.CrossEntropyLoss()
    
    # Different loss functions for different tasks
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    
    model.train()
    losses = []
    task_accuracies = {i: [] for i in range(len(task_types))}
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        total_samples = 0
        epoch_task_correct = {i: 0 for i in range(len(task_types))}
        epoch_task_total = {i: 0 for i in range(len(task_types))}
        
        # Iterate through all task data loaders
        for task_id, loader in train_loaders.items():
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                
                # Create task labels
                task_labels = torch.full((batch_x.size(0),), task_id, dtype=torch.long)
                
                # Forward pass
                output, task_probs = model(batch_x, task_label=task_labels)
                
                # Task classification loss
                task_loss = task_criterion(task_probs, task_labels)
                
                # Main task loss - use the known output size for each task
                if task_types[task_id] == 'classification':
                    # Use the predefined output size (number of classes) for this task
                    n_classes = output_sizes[task_id]
                    main_loss = classification_criterion(output[:, :n_classes], batch_y)
                else:
                    # For regression, use the actual target size
                    target_size = batch_y.size(1)
                    main_loss = regression_criterion(output[:, :target_size], batch_y)
                
                # Combined loss
                total_loss = main_loss + 0.2 * task_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)
                
                # Track task classification accuracy
                task_pred = torch.argmax(task_probs, dim=1)
                epoch_task_correct[task_id] += (task_pred == task_labels).sum().item()
                epoch_task_total[task_id] += batch_x.size(0)
        
        avg_loss = epoch_loss / total_samples
        losses.append(avg_loss)
        
        # Calculate task accuracies
        for task_id in range(len(task_types)):
            if epoch_task_total[task_id] > 0:
                acc = epoch_task_correct[task_id] / epoch_task_total[task_id]
                task_accuracies[task_id].append(acc)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
            for task_id in range(len(task_types)):
                if epoch_task_total[task_id] > 0:
                    acc = epoch_task_correct[task_id] / epoch_task_total[task_id]
                    print(f'  Task {task_id} Classification Acc: {acc:.4f}')
    
    return losses, task_accuracies

# Example usage
if __name__ == "__main__":
    print("Creating 10 diverse datasets...")
    datasets, task_types = create_diverse_datasets(n_samples=2000, input_size=100)
    
    # Determine output sizes for each task
    output_sizes = []
    for task_id in range(10):
        if task_types[task_id] == 'classification':
            # Get the actual number of classes in the full dataset
            output_sizes.append(len(torch.unique(datasets[task_id][1])))
        else:
            output_sizes.append(datasets[task_id][1].size(1))
    
    print(f"Task types: {task_types}")
    print(f"Output sizes: {output_sizes}")
    
    # Create data loaders
    train_loaders = {}
    for task_id in range(10):
        X, y = datasets[task_id]
        train_loaders[task_id] = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    
    # Initialize model with larger cache
    model = DynamicWeightNetwork(
        input_size=100,
        hidden_sizes=[256, 128, 64],
        output_sizes=output_sizes,
        num_tasks=10,
        cache_size=25  # Larger cache for more complex task switching
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture:")
    print(f"Total parameters: {total_params:,}")
    print(f"Cache size per task-layer: 25")
    print(f"Tasks: {len(task_types)}")
    
    print("\nTraining multi-task network...")
    losses, task_accuracies = train_multi_task_network(
        model, train_loaders, task_types, output_sizes, num_epochs=100, lr=0.001
    )
    
    # Comprehensive testing
    print("\n" + "="*60)
    print("COMPREHENSIVE TESTING RESULTS")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        for task_id in range(10):
            X_test, y_test = datasets[task_id]
            test_data = X_test[:200]  # Test on 200 samples
            
            # Test task classification
            output, task_probs = model(test_data)
            predicted_tasks = torch.argmax(task_probs, dim=1)
            task_accuracy = (predicted_tasks == task_id).float().mean()
            
            print(f"\nTask {task_id} ({task_types[task_id]}):")
            print(f"  Task Classification Accuracy: {task_accuracy:.4f}")
            print(f"  Predicted tasks sample: {predicted_tasks[:10].tolist()}")
            
            if task_types[task_id] == 'classification':
                pred_classes = torch.argmax(output[:, :output_sizes[task_id]], dim=1)
                true_classes = y_test[:200]
                main_accuracy = (pred_classes == true_classes).float().mean()
                print(f"  Main Task Accuracy: {main_accuracy:.4f}")
            else:
                pred_values = output[:, :output_sizes[task_id]]
                true_values = y_test[:200]
                mse = F.mse_loss(pred_values, true_values)
                print(f"  Main Task MSE: {mse:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Task classification accuracies
    for task_id, accs in task_accuracies.items():
        if len(accs) > 0:
            axes[0, 1].plot(accs, label=f'Task {task_id}')
    axes[0, 1].set_title('Task Classification Accuracies')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True)
    
    # Final task classification accuracy heatmap
    final_accs = [task_accuracies[i][-1] if len(task_accuracies[i]) > 0 else 0 for i in range(10)]
    axes[1, 0].bar(range(10), final_accs)
    axes[1, 0].set_title('Final Task Classification Accuracy')
    axes[1, 0].set_xlabel('Task ID')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].grid(True, alpha=0.3)
    
    # Task type distribution
    task_type_counts = {}
    for task_type in task_types.values():
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
    
    axes[1, 1].pie(task_type_counts.values(), labels=task_type_counts.keys(), autopct='%1.1f%%')
    axes[1, 1].set_title('Task Type Distribution')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🎉 SUCCESS! Dynamic Weight Network with 10 tasks trained successfully!")
    print(f"📊 Model efficiently handles {len(task_types)} different tasks")
    print(f"🧠 Cache-based architecture with {total_params:,} parameters")
    print(f"🎯 Average task classification accuracy: {np.mean(final_accs):.4f}")
