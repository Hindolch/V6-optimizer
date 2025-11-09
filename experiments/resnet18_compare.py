import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys, os
import torch.distributed as dist

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizers.adamw_wrapper import AdamWWrapper
from optimizers.biostatis import BiostatisV6
from gpu_monitor import start_gpu_monitor

# ---------------------
# Training utility
# ---------------------
def train_and_eval(optimizer_name, optimizer_class, trainloader, testloader, device, epochs=3, **optimizer_kwargs):
    import time
    # ResNet18 backbone
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Special handling for Muon optimizer
    if optimizer_name == "Muon":
        optimizer = create_muon_optimizer(model, **optimizer_kwargs)  # pyright: ignore[reportUndefinedVariable]
    else:
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    train_losses, test_accs = [], []
    epoch_times = []
    total_start = time.perf_counter()

    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0
        # for accurate GPU timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        num_train_samples = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Provide closure for optimizers that need it
            def closure():
                with torch.enable_grad():
                    out = model(inputs)
                    return criterion(out, labels)

            # DynamoRayleigh needs closure
            if optimizer_name == "DynamoV2":
                optimizer.step(closure=closure)
            else:
                optimizer.step()
            
            running_loss += loss.item()
            num_train_samples += inputs.size(0)

        # end epoch timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        train_losses.append(running_loss / len(trainloader))

        # test
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accs.append(100 * correct / total)
        imgs_per_sec = num_train_samples / epoch_time if epoch_time > 0 else float('inf')
        print(f"{optimizer_name} | Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Acc={test_accs[-1]:.2f}% | Time={epoch_time:.2f}s | Throughput={imgs_per_sec:.1f} img/s")
    total_time = time.perf_counter() - total_start
    timing_stats = {
        "epoch_times": epoch_times,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times) if epoch_times else 0.0,
        "total_time": total_time,
    }
    return train_losses, test_accs, model, timing_stats

# ---------------------
# Spectral analysis
# ---------------------
def compute_singular_values(model, layer_name="fc.weight"):
    param = dict(model.named_parameters())[layer_name].detach().cpu()
    param_2d = param.reshape(param.shape[0], -1)
    _, S, _ = torch.linalg.svd(param_2d, full_matrices=False)
    return S.cpu().numpy()

# ---------------------
# Main
# ---------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize distributed training ONLY for Muon if needed
    # For single GPU, we can use Muon without distributed training
    # by using the simpler Muon class instead of MuonWithAuxAdam

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    results = {}
    singular_spectra = {}

    # Define optimizers with proper hyperparameters
    optimizers_config = [
        # AdamW baseline
        ("AdamW", AdamWWrapper, {
            "lr": 3e-4, 
            "weight_decay": 1e-2
        }),
    
        ("BiostatisV6", BiostatisV6,{
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "homeo_rate": 0.5,
            "coherence_target": 0.8,
            "energy_target": 1e-4,
            "lambda_energy": 0.1
        }),

               
    ]

    for name, opt_class, opt_kwargs in optimizers_config:
        print(f"\n===== Training with {name} (ResNet18) =====")
        print(f"Hyperparameters: {opt_kwargs}")
        
        try:
            # Start GPU monitor
            gpu_thread = start_gpu_monitor(name, interval=5)

            losses, accs, model, timing = train_and_eval(name, opt_class, trainloader, testloader, device, epochs=5, **opt_kwargs)
            results[name] = (losses, accs, timing)

            # Save checkpoint and compute singular spectrum
            os.makedirs("results/cifar10_resnet", exist_ok=True)
            ckpt_path = os.path.join("results", "cifar10_resnet", f"{name.lower()}_resnet18.pth")
            torch.save(model.state_dict(), ckpt_path)

            singular_spectra[name] = compute_singular_values(model, layer_name="fc.weight")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Test Accuracy
    for name, (losses, accs, timing) in results.items():
        axes[0, 0].plot(accs, label=name, marker='o', linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Test Accuracy (%)")
    axes[0, 0].set_title("CIFAR-10 ResNet18 Test Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    for name, (losses, accs, timing) in results.items():
        axes[0, 1].plot(losses, label=name, marker='s', linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Training Loss")
    axes[0, 1].set_title("CIFAR-10 ResNet18 Training Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Singular Spectrum
    for name, spectrum in singular_spectra.items():
        axes[1, 0].plot(spectrum, label=name, linewidth=2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Singular Value")
    axes[1, 0].set_title("Singular Spectrum of fc.weight")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Final Performance Summary
    final_accs = [results[name][1][-1] for name in results.keys()]
    bars = axes[1, 1].bar(results.keys(), final_accs, alpha=0.7)
    axes[1, 1].set_ylabel("Final Test Accuracy (%)")
    axes[1, 1].set_title("Final Performance Comparison")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("results/cifar10_resnet/comprehensive_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for name, (losses, accs, timing) in results.items():
        final_loss = losses[-1]
        final_acc = accs[-1]
        best_acc = max(accs)
        best_epoch = accs.index(best_acc) + 1
        avg_epoch = timing.get("avg_epoch_time", 0.0)
        total_time = timing.get("total_time", 0.0)
        print(f"{name:15} | Final: {final_acc:6.2f}% | Best: {best_acc:6.2f}% (Epoch {best_epoch}) | Avg epoch: {avg_epoch:6.2f}s | Total: {total_time:7.2f}s")

    # Speed comparison plot (avg epoch time)
    try:
        speed_names = list(results.keys())
        avg_epoch_times = [results[n][2]["avg_epoch_time"] for n in speed_names]
        plt.figure(figsize=(6,4))
        plt.bar(speed_names, avg_epoch_times, alpha=0.8)
        plt.ylabel("Avg Epoch Time (s)")
        plt.title("Optimizer Speed Comparison (ResNet18 CIFAR-10)")
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        os.makedirs("results/cifar10_resnet", exist_ok=True)
        plt.savefig("results/cifar10_resnet/speed_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not create speed plot: {e}")
    
    print(f"{'='*60}")
    
    # Analysis of singular spectra
    print("\nSINGULAR SPECTRUM ANALYSIS:")
    for name, spectrum in singular_spectra.items():
        top_5_ratio = spectrum[:5].sum() / spectrum.sum()
        effective_rank = (spectrum.sum()**2) / (spectrum**2).sum()
        print(f"{name:15} | Top-5 concentration: {top_5_ratio:.3f} | Effective rank: {effective_rank:.2f}")


if __name__ == "__main__":
    print("Make sure you have installed:")
    print("  pip install lion-pytorch")
    print("  pip install git+https://github.com/KellerJordan/Muon")
    print()
    main()