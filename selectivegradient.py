import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

class SelectiveGradientConv2d(nn.Module):
    """
    Convolutional layer with selective gradient propagation.
    Uses mean gradient selection and standard deviation scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SelectiveGradientConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        
        # Track gradient statistics
        self.gradient_history = []
        self.selected_paths = set()
        self.layer_id = id(self)
        
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
    
    def selective_backward(self, grad_input):
        """Apply selective gradient to input gradients"""
        grad_flat = grad_input.view(grad_input.size(0), -1)
        mean_grad = torch.mean(grad_flat, dim=1)
        std_grad = torch.std(grad_flat, dim=1)

        mean_magnitude = torch.mean(torch.abs(mean_grad))
        selected_indices = torch.abs(mean_grad) >= mean_magnitude

        if std_grad.mean() > 0:
            std_multiplier = torch.abs(mean_grad - mean_grad.mean()) / (std_grad.mean() + 1e-8)
        else:
            std_multiplier = torch.ones_like(mean_grad)

        selective_mask = selected_indices.float().unsqueeze(-1).expand_as(grad_flat)
        std_scale = std_multiplier.unsqueeze(-1).expand_as(grad_flat)

        modified_grad = grad_flat * selective_mask * std_scale
        return modified_grad.view_as(grad_input)

class SelectiveGradientCNN(nn.Module):
    """
    CNN with selective gradient propagation layers
    """
    def __init__(self, num_classes=10):
        super(SelectiveGradientCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = SelectiveGradientConv2d(1, 32, 3, padding=1)
        self.conv2 = SelectiveGradientConv2d(32, 64, 3, padding=1)
        self.conv3 = SelectiveGradientConv2d(64, 128, 3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Hook for gradient modification
        self.register_backward_hooks()
        
    def register_backward_hooks(self):
        def conv_backward_hook(module, grad_input, grad_output):
            if isinstance(module, SelectiveGradientConv2d):
                # Modify only the input gradient (first element of grad_input)
                modified_grad = module.selective_backward(grad_input[0])
                return (modified_grad,) + grad_input[1:]
            return grad_input
        
        # Register hooks only for backward pass
        self.conv1.register_backward_hook(conv_backward_hook)
        self.conv2.register_backward_hook(conv_backward_hook)
        self.conv3.register_backward_hook(conv_backward_hook)
    
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third conv block
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class StandardCNN(nn.Module):
    """
    Standard CNN for comparison
    """
    def __init__(self, num_classes=10):
        super(StandardCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, test_loader, num_epochs=5, lr=0.001):
    """Train the model and return training statistics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    training_times = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_time = time.time() - start_time
        training_times.append(epoch_time)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Testing phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'training_times': training_times
    }

def plot_gradient_statistics(model):
    """Plot gradient statistics for selective gradient model"""
    if not isinstance(model, SelectiveGradientCNN):
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    layers = [model.conv1, model.conv2, model.conv3]
    layer_names = ['Conv1', 'Conv2', 'Conv3']
    
    for i, (layer, name) in enumerate(zip(layers, layer_names)):
        if len(layer.gradient_history) > 0:
            # Plot selection ratios
            selection_ratios = [h['selected_ratio'] for h in layer.gradient_history]
            axes[0, 0].plot(selection_ratios, label=name)
            
            # Plot mean gradient magnitudes
            mean_grad_mags = [np.mean(np.abs(h['mean_grad'])) for h in layer.gradient_history]
            axes[0, 1].plot(mean_grad_mags, label=name)
            
            # Plot standard deviations
            std_grads = [np.mean(h['std_grad']) for h in layer.gradient_history]
            axes[1, 0].plot(std_grads, label=name)
            
            # Plot gradient distribution for last epoch
            if len(layer.gradient_history) > 0:
                last_grads = layer.gradient_history[-1]['mean_grad']
                axes[1, 1].hist(last_grads, alpha=0.7, bins=20, label=name)
    
    axes[0, 0].set_title('Selection Ratios Over Time')
    axes[0, 0].set_xlabel('Update Step')
    axes[0, 0].set_ylabel('Selection Ratio')
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Mean Gradient Magnitudes')
    axes[0, 1].set_xlabel('Update Step')
    axes[0, 1].set_ylabel('Mean |Gradient|')
    axes[0, 1].legend()
    
    axes[1, 0].set_title('Gradient Standard Deviations')
    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Std Gradient')
    axes[1, 0].legend()
    
    axes[1, 1].set_title('Final Gradient Distribution')
    axes[1, 1].set_xlabel('Gradient Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("Training Selective Gradient CNN...")
    selective_model = SelectiveGradientCNN()
    selective_stats = train_model(selective_model, train_loader, test_loader, num_epochs=5)
    
    print("\nTraining Standard CNN...")
    standard_model = StandardCNN()
    standard_stats = train_model(standard_model, train_loader, test_loader, num_epochs=5)
    
    # Compare results
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    print(f"Selective Gradient CNN - Final Test Accuracy: {selective_stats['test_accuracies'][-1]:.2f}%")
    print(f"Standard CNN - Final Test Accuracy: {standard_stats['test_accuracies'][-1]:.2f}%")
    
    print(f"\nSelective Gradient CNN - Avg Training Time: {np.mean(selective_stats['training_times']):.2f}s/epoch")
    print(f"Standard CNN - Avg Training Time: {np.mean(standard_stats['training_times']):.2f}s/epoch")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(1, len(selective_stats['train_losses']) + 1)
    
    # Training loss
    axes[0].plot(epochs, selective_stats['train_losses'], 'b-', label='Selective Gradient')
    axes[0].plot(epochs, standard_stats['train_losses'], 'r-', label='Standard')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Test accuracy
    axes[1].plot(epochs, selective_stats['test_accuracies'], 'b-', label='Selective Gradient')
    axes[1].plot(epochs, standard_stats['test_accuracies'], 'r-', label='Standard')
    axes[1].set_title('Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    
    # Training time
    axes[2].plot(epochs, selective_stats['training_times'], 'b-', label='Selective Gradient')
    axes[2].plot(epochs, standard_stats['training_times'], 'r-', label='Standard')
    axes[2].set_title('Training Time per Epoch')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Time (seconds)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot gradient statistics for selective model
    plot_gradient_statistics(selective_model)
    
    return selective_model, standard_model, selective_stats, standard_stats

if __name__ == "__main__":
    selective_model, standard_model, selective_stats, standard_stats = main()
