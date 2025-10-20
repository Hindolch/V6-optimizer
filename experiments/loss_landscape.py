import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizers.adamw_wrapper import AdamWWrapper
from optimizers.dynamo import BiostatisV4, BiostatisV5
from torch.optim import RAdam

class LossLandscapeVisualizer:
    """
    Visualizes loss landscapes for models trained with different optimizers.
    Creates 2D and 3D plots showing how loss varies around the final parameters.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.setup_data()
        
    def setup_data(self):
        """Load CIFAR-10 test data for loss computation."""
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=256, shuffle=False, num_workers=2
        )
        
    def load_model_checkpoint(self, checkpoint_path):
        """Load a trained model from checkpoint."""
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
        
    def compute_loss_at_point(self, model, direction1, direction2, alpha, beta, criterion):
        """Compute loss at a point (alpha, beta) in the 2D parameter space."""
        # Get original parameters
        original_params = [p.clone() for p in model.parameters()]
        
        # Move parameters to the specified point
        for i, param in enumerate(model.parameters()):
            param.data = original_params[i] + alpha * direction1[i] + beta * direction2[i]
            
        # Compute loss on test set
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
                
        # Restore original parameters
        for i, param in enumerate(model.parameters()):
            param.data = original_params[i]
            
        return total_loss / num_batches
        
    def generate_random_directions(self, model):
        """Generate two random directions in parameter space."""
        params = list(model.parameters())
        
        # Generate random directions
        direction1 = []
        direction2 = []
        
        for param in params:
            # Random direction with same shape as parameter
            rand1 = torch.randn_like(param)
            rand2 = torch.randn_like(param)
            
            # Normalize to have unit norm
            rand1 = rand1 / torch.norm(rand1)
            rand2 = rand2 / torch.norm(rand2)
            
            direction1.append(rand1)
            direction2.append(rand2)
            
        return direction1, direction2
        
    def plot_loss_landscape_2d(self, model, optimizer_name, direction1, direction2, 
                              alpha_range=(-1, 1), beta_range=(-1, 1), resolution=20):
        """Create 2D loss landscape plot."""
        criterion = nn.CrossEntropyLoss()
        
        # Create grid
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], resolution)
        beta_values = np.linspace(beta_range[0], beta_range[1], resolution)
        Alpha, Beta = np.meshgrid(alpha_values, beta_values)
        
        # Compute loss at each grid point
        Loss = np.zeros_like(Alpha)
        
        print(f"Computing loss landscape for {optimizer_name}...")
        for i in range(resolution):
            for j in range(resolution):
                alpha = alpha_values[i]
                beta = beta_values[j]
                loss = self.compute_loss_at_point(model, direction1, direction2, alpha, beta, criterion)
                Loss[j, i] = loss
                print(f"Progress: {i*resolution + j + 1}/{resolution**2} - Loss: {loss:.4f}")
        
        return Alpha, Beta, Loss
        
    def plot_loss_landscape_3d(self, Alpha, Beta, Loss, optimizer_name, save_path):
        """Create 3D loss landscape plot."""
        fig = plt.figure(figsize=(12, 8))
        
        # 3D surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(Alpha, Beta, Loss, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('Beta')
        ax1.set_zlabel('Loss')
        ax1.set_title(f'{optimizer_name} - 3D Loss Landscape')
        plt.colorbar(surf, ax=ax1, shrink=0.5)
        
        # 2D contour plot
        ax2 = fig.add_subplot(122)
        contour = ax2.contour(Alpha, Beta, Loss, levels=20, colors='black', alpha=0.6)
        contourf = ax2.contourf(Alpha, Beta, Loss, levels=20, cmap='viridis', alpha=0.8)
        ax2.set_xlabel('Alpha')
        ax2.set_ylabel('Beta')
        ax2.set_title(f'{optimizer_name} - 2D Contour')
        plt.colorbar(contourf, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_comparison_landscapes(self, landscapes, save_path):
        """Plot comparison of multiple optimizers' loss landscapes."""
        n_optimizers = len(landscapes)
        fig, axes = plt.subplots(2, n_optimizers, figsize=(5*n_optimizers, 10))
        
        if n_optimizers == 1:
            axes = axes.reshape(2, 1)
            
        for i, (optimizer_name, (Alpha, Beta, Loss)) in enumerate(landscapes.items()):
            # 3D surface
            ax1 = axes[0, i]
            ax1.remove()
            ax1 = fig.add_subplot(2, n_optimizers, i+1, projection='3d')
            surf = ax1.plot_surface(Alpha, Beta, Loss, cmap='viridis', alpha=0.8)
            ax1.set_xlabel('Alpha')
            ax1.set_ylabel('Beta')
            ax1.set_zlabel('Loss')
            ax1.set_title(f'{optimizer_name} - 3D')
            
            # 2D contour
            ax2 = axes[1, i]
            contourf = ax2.contourf(Alpha, Beta, Loss, levels=20, cmap='viridis', alpha=0.8)
            ax2.contour(Alpha, Beta, Loss, levels=20, colors='black', alpha=0.6)
            ax2.set_xlabel('Alpha')
            ax2.set_ylabel('Beta')
            ax2.set_title(f'{optimizer_name} - 2D')
            
            # Add colorbar
            plt.colorbar(contourf, ax=ax2)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_landscape_properties(self, landscapes):
        """Analyze and compare landscape properties."""
        print("\n" + "="*60)
        print("LOSS LANDSCAPE ANALYSIS")
        print("="*60)
        
        for optimizer_name, (Alpha, Beta, Loss) in landscapes.items():
            # Basic statistics
            min_loss = np.min(Loss)
            max_loss = np.max(Loss)
            mean_loss = np.mean(Loss)
            std_loss = np.std(Loss)
            
            # Find minimum point
            min_idx = np.unravel_index(np.argmin(Loss), Loss.shape)
            min_alpha = Alpha[min_idx]
            min_beta = Beta[min_idx]
            
            # Compute gradients (approximate)
            grad_alpha = np.gradient(Loss, axis=1)
            grad_beta = np.gradient(Loss, axis=0)
            grad_magnitude = np.sqrt(grad_alpha**2 + grad_beta**2)
            max_gradient = np.max(grad_magnitude)
            mean_gradient = np.mean(grad_magnitude)
            
            # Roughness (standard deviation of gradients)
            roughness = np.std(grad_magnitude)
            
            print(f"\n{optimizer_name}:")
            print(f"  Loss range: {min_loss:.4f} - {max_loss:.4f}")
            print(f"  Mean loss: {mean_loss:.4f} ± {std_loss:.4f}")
            print(f"  Minimum at: α={min_alpha:.3f}, β={min_beta:.3f}")
            print(f"  Max gradient: {max_gradient:.4f}")
            print(f"  Mean gradient: {mean_gradient:.4f}")
            print(f"  Roughness: {roughness:.4f}")
            
    def create_landscape_comparison(self, checkpoint_dir="results/cifar10_resnet"):
        """Main function to create loss landscape comparison."""
        # Define optimizers and their checkpoints
        optimizers = {
            "AdamW": os.path.join(checkpoint_dir, "adamw_resnet18.pth"),
            "BiostatisV4": os.path.join(checkpoint_dir, "biostatisv4_resnet18.pth"),
            "BiostatisV5": os.path.join(checkpoint_dir, "biostatisv5_resnet18.pth"),
            "RAdam": os.path.join(checkpoint_dir, "radam_resnet18.pth"),
        }
        
        # Check which checkpoints exist
        available_optimizers = {}
        for name, path in optimizers.items():
            if os.path.exists(path):
                available_optimizers[name] = path
            else:
                print(f"Warning: Checkpoint not found for {name}: {path}")
        
        if not available_optimizers:
            print("No checkpoints found! Please train models first.")
            return
            
        print(f"Found checkpoints for: {list(available_optimizers.keys())}")
        
        # Create output directory
        output_dir = "results/loss_landscapes"
        os.makedirs(output_dir, exist_ok=True)
        
        landscapes = {}
        
        # Generate landscapes for each optimizer
        for optimizer_name, checkpoint_path in available_optimizers.items():
            print(f"\nProcessing {optimizer_name}...")
            
            # Load model
            model = self.load_model_checkpoint(checkpoint_path)
            
            # Generate random directions (same for all optimizers for fair comparison)
            if 'direction1' not in locals():
                direction1, direction2 = self.generate_random_directions(model)
            
            # Compute loss landscape
            Alpha, Beta, Loss = self.plot_loss_landscape_2d(
                model, optimizer_name, direction1, direction2,
                alpha_range=(-0.5, 0.5), beta_range=(-0.5, 0.5), resolution=15
            )
            
            landscapes[optimizer_name] = (Alpha, Beta, Loss)
            
            # Save individual landscape
            individual_path = os.path.join(output_dir, f"{optimizer_name.lower()}_landscape.png")
            self.plot_loss_landscape_3d(Alpha, Beta, Loss, optimizer_name, individual_path)
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # Create comparison plot
        if len(landscapes) > 1:
            comparison_path = os.path.join(output_dir, "landscape_comparison.png")
            self.plot_comparison_landscapes(landscapes, comparison_path)
            
        # Analyze landscape properties
        self.analyze_landscape_properties(landscapes)
        
        print(f"\nLoss landscape plots saved to: {output_dir}")
        
        return landscapes

def main():
    """Main function to run loss landscape visualization."""
    print("Loss Landscape Visualization")
    print("="*40)
    
    visualizer = LossLandscapeVisualizer()
    landscapes = visualizer.create_landscape_comparison()
    
    if landscapes:
        print("\nVisualization complete!")
        print("Check the 'results/loss_landscapes' directory for plots.")
    else:
        print("No landscapes were generated. Check if model checkpoints exist.")

if __name__ == "__main__":
    main()
