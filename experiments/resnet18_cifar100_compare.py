# -*- coding: utf-8 -*-
"""
CIFAR-100 ResNet18 优化器对比实验 (单文件版)

本脚本将多个自定义优化器 (Dynamo, Muon, Lion) 与标准优化器 (AdamW, RAdam)
在 CIFAR-100 数据集上使用 ResNet18 模型进行性能对比。

所有必要的代码，包括优化器实现和工具函数，都已整合到此文件中，无需额外安装库
(除了 torch, torchvision, matplotlib)。

主要功能:
1. 在 CIFAR-100 上训练和评估多个优化器。
2. 监控训练过程中的 GPU 使用情况。
3. 绘制并保存训练损失、测试准确率、最终性能和权重奇异值谱的对比图。
4. 输出详细的性能总结和奇异谱分析。
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys
import time
import math
import subprocess
import threading
from typing import Tuple, Callable
from torch.optim.optimizer import Optimizer

# Ensure project root is importable for optimizer modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizers.biostatis import BiostatisV6



def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

# ---------------------
# AdamW Wrapper (from optimizers/adamw_wrapper.py)
# ---------------------
def AdamWWrapper(params, **kwargs):
    return torch.optim.AdamW(params, **kwargs)


# ==============================================================================
# §2. UTILITY FUNCTIONS
# ==============================================================================

# ---------------------
# GPU Monitor (from gpu_monitor.py)
# ---------------------
def _gpu_worker(logfile, interval=5):
    with open(logfile, "w") as f:
        f.write("time, gpu_util, mem_used(MB), mem_total(MB)\n")
        while True:
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"]
                )
                line = result.decode("utf-8").strip()
                gpu_util, mem_used, mem_total = [x.strip() for x in line.split(",")]
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp}, {gpu_util}, {mem_used}, {mem_total}\n")
                f.flush()
            except Exception:
                pass # Suppress errors if nvidia-smi is not found or fails
            time.sleep(interval)

def start_gpu_monitor(optimizer_name, logdir="results/gpu_logs", interval=5):
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f"gpu_log_{optimizer_name.lower()}.csv")
    t = threading.Thread(target=_gpu_worker, args=(logfile, interval), daemon=True)
    t.start()
    print(f"[GPU Monitor] Logging to {logfile} every {interval}s")
    return t


# ---------------------
# Training & Evaluation
# ---------------------
def train_and_eval(optimizer_name, optimizer_class, trainloader, testloader, device, epochs=3, **optimizer_kwargs):
    model = torchvision.models.resnet18(weights=None)
    # Change output layer for CIFAR-100
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    train_losses, test_accs = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(trainloader))

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
        print(f"{optimizer_name} | Epoch {epoch+1:02d}/{epochs}: Loss={train_losses[-1]:.4f}, Acc={test_accs[-1]:.2f}%")

    return train_losses, test_accs, model

# ---------------------
# Spectral Analysis
# ---------------------
def compute_singular_values(model, layer_name="fc.weight"):
    with torch.no_grad():
        param = dict(model.named_parameters())[layer_name].detach().cpu()
        param_2d = param.reshape(param.shape[0], -1)
        _, S, _ = torch.linalg.svd(param_2d, full_matrices=False)
    return S.cpu().numpy()

# ==============================================================================
# §3. MAIN EXPERIMENT
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-100 specific transforms with standard normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    results = {}
    singular_spectra = {}
    
    # NOTE: Hyperparameters are kept from the original script. They may not be optimal for CIFAR-100.
    optimizers_config = [
        ("AdamW", AdamWWrapper, {"lr": 3e-4, "weight_decay": 1e-2}),
        ("BiostatisV6", BiostatisV6, {
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "homeo_rate": 0.5,
            "coherence_target": 0.8,
            "energy_target": 1e-4,
            "lambda_energy": 0.1
        }),
    ]

    for name, opt_class, opt_kwargs in optimizers_config:
        print(f"\n===== Training with {name} (ResNet18 on CIFAR-100) =====")
        print(f"Hyperparameters: {opt_kwargs}")
        try:
            gpu_thread = start_gpu_monitor(name, interval=5)
            # Increased epochs for the more complex CIFAR-100 task
            losses, accs, model = train_and_eval(name, opt_class, trainloader, testloader, device, epochs=20, **opt_kwargs)
            results[name] = (losses, accs)

            output_dir = "results/cifar100_resnet"
            os.makedirs(output_dir, exist_ok=True)
            ckpt_path = os.path.join(output_dir, f"{name.lower()}_resnet18.pth")
            torch.save(model.state_dict(), ckpt_path)

            singular_spectra[name] = compute_singular_values(model, layer_name="fc.weight")
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("ResNet18 Optimizer Comparison on CIFAR-100", fontsize=16)
    
    for name, (losses, accs) in results.items():
        axes[0, 0].plot(accs, label=name, marker='o', markersize=4, linewidth=2)
    axes[0, 0].set_xlabel("Epoch"), axes[0, 0].set_ylabel("Test Accuracy (%)")
    axes[0, 0].set_title("Test Accuracy"), axes[0, 0].legend(), axes[0, 0].grid(True, alpha=0.4)
    
    for name, (losses, accs) in results.items():
        axes[0, 1].plot(losses, label=name, marker='s', markersize=4, linewidth=2)
    axes[0, 1].set_xlabel("Epoch"), axes[0, 1].set_ylabel("Training Loss")
    axes[0, 1].set_title("Training Loss"), axes[0, 1].legend(), axes[0, 1].grid(True, alpha=0.4)
    
    for name, spectrum in singular_spectra.items():
        axes[1, 0].plot(spectrum, label=name, linewidth=2)
    axes[1, 0].set_yscale("log"), axes[1, 0].set_xlabel("Index"), axes[1, 0].set_ylabel("Singular Value")
    axes[1, 0].set_title("Singular Spectrum of Final Layer"), axes[1, 0].legend(), axes[1, 0].grid(True, alpha=0.4)
    
    final_accs = {name: res[1][-1] for name, res in results.items()}
    sorted_names = sorted(final_accs, key=final_accs.get, reverse=True)
    sorted_accs = [final_accs[name] for name in sorted_names]
    bars = axes[1, 1].bar(sorted_names, sorted_accs, alpha=0.8)
    axes[1, 1].set_ylabel("Final Test Accuracy (%)"), axes[1, 1].set_title("Final Performance")
    axes[1, 1].grid(True, axis='y', alpha=0.4)
    axes[1, 1].tick_params(axis='x', rotation=45)
    for label in axes[1, 1].get_xticklabels():
        label.set_ha('right')
    for bar, acc in zip(bars, sorted_accs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "results/cifar100_resnet/comprehensive_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {save_path}")
    plt.show()

    # Final Summary
    print(f"\n{'='*60}\nFINAL RESULTS SUMMARY (CIFAR-100)\n{'='*60}")
    for name in sorted_names:
        losses, accs = results[name]
        print(f"{name:22} | Final Acc: {accs[-1]:6.2f}% | Best Acc: {max(accs):6.2f}% (Epoch {accs.index(max(accs)) + 1})")
    print(f"{'='*60}")
    
    print("\nSINGULAR SPECTRUM ANALYSIS:")
    for name in sorted_names:
        spectrum = singular_spectra[name]
        top_5_ratio = spectrum[:5].sum() / spectrum.sum()
        effective_rank = (spectrum.sum()**2) / (spectrum**2).sum()
        print(f"{name:22} | Top-5 concentration: {top_5_ratio:.3f} | Effective rank: {effective_rank:.2f}")

if __name__ == "__main__":
    main()
