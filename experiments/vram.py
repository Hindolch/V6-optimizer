import torch
import gc
import torchvision
from torch.optim import AdamW

# Import your optimizers (adjust path as needed)
from optimizers.dynamo import BiostatisV5_1, BiostatisV5, BiostatisV6

def get_mem():
    """Get current CUDA memory in MB"""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2

def detailed_vram_analysis(model, optimizer_cls, **opt_kwargs):
    """Measure VRAM breakdown with detailed stages"""
    torch.cuda.empty_cache()
    gc.collect()
    
    measurements = {}
    
    # Stage 0: Empty GPU
    measurements['0_empty'] = get_mem()
    
    # Stage 1: Model weights only
    model = model.cuda()
    measurements['1_weights'] = get_mem()
    
    # Stage 2: After creating optimizer (before any step)
    opt = optimizer_cls(model.parameters(), **opt_kwargs)
    measurements['2_optimizer_init'] = get_mem()
    
    # Stage 3: After allocating gradients
    x = torch.randn(16, 3, 32, 32, device="cuda")
    y = torch.randint(0, 100, (16,), device="cuda")
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    measurements['3_after_backward'] = get_mem()
    
    # Stage 4: After first optimizer step (state allocated)
    opt.step()
    measurements['4_after_step'] = get_mem()
    
    # Stage 5: After second step (ensure all state is warm)
    opt.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    opt.step()
    measurements['5_steady_state'] = get_mem()
    
    # Compute deltas
    deltas = {
        'weights_only': measurements['1_weights'] - measurements['0_empty'],
        'optimizer_overhead_init': measurements['2_optimizer_init'] - measurements['1_weights'],
        'gradients': measurements['3_after_backward'] - measurements['2_optimizer_init'],
        'optimizer_state': measurements['4_after_step'] - measurements['3_after_backward'],
        'total_steady': measurements['5_steady_state'] - measurements['0_empty']
    }
    
    return measurements, deltas

def count_parameters(model):
    """Count total parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def theoretical_memory(param_count, optimizer_name):
    """Calculate theoretical memory usage"""
    bytes_per_param = 4  # float32
    mb = param_count * bytes_per_param / 1024**2
    
    if optimizer_name == "AdamW":
        # weights + gradients + m_t + v_t
        return {
            'weights': mb,
            'gradients': mb,
            'state': 2 * mb,  # m_t + v_t
            'total': 4 * mb
        }
    elif optimizer_name in ["BiostatisV5", "BiostatisV5_1", "BiostatisV6"]:
        # weights + gradients + m_t + v_t + 3×memory_emas
        return {
            'weights': mb,
            'gradients': mb,
            'state': 5 * mb,  # m_t + v_t + 3×EMA
            'total': 7 * mb
        }
    return {}

# =============================================================================
# Run comparison
# =============================================================================

print("="*80)
print("DETAILED VRAM ANALYSIS: AdamW vs BiostatisV5 vs BiostatisV5_1 vs BiostatisV6")
print("="*80)

model = torchvision.models.resnet18(num_classes=100)
total_params, trainable_params = count_parameters(model)
print(f"\nModel: ResNet18")
print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"Trainable parameters: {trainable_params:,}")

results = {}
optimizers = [
    ('AdamW', AdamW, {'lr': 1e-3}),
    ('BiostatisV5', BiostatisV5, {'lr': 1e-3}),
    ('BiostatisV5_1', BiostatisV5_1, {'lr': 1e-3}),
    ('BiostatisV6', BiostatisV6, {'lr': 1e-3}),
]

for name, opt_cls, kwargs in optimizers:
    print(f"\n{'-'*80}")
    print(f"Testing: {name}")
    print(f"{'-'*80}")
    
    # Fresh model for each test
    test_model = torchvision.models.resnet18(num_classes=100)
    measurements, deltas = detailed_vram_analysis(test_model, opt_cls, **kwargs)
    results[name] = deltas
    
    # Print detailed breakdown
    print(f"\nMeasured VRAM Breakdown:")
    print(f"  Weights only:              {deltas['weights_only']:>8.2f} MB")
    print(f"  Optimizer init overhead:   {deltas['optimizer_overhead_init']:>8.2f} MB")
    print(f"  Gradients:                 {deltas['gradients']:>8.2f} MB")
    print(f"  Optimizer state (after step): {deltas['optimizer_state']:>8.2f} MB")
    print(f"  {'─'*50}")
    print(f"  Total steady state:        {deltas['total_steady']:>8.2f} MB")
    
    # Theoretical calculation
    theoretical = theoretical_memory(total_params, name)
    if theoretical:
        print(f"\nTheoretical VRAM:")
        print(f"  Weights:                   {theoretical['weights']:>8.2f} MB")
        print(f"  Gradients:                 {theoretical['gradients']:>8.2f} MB")
        print(f"  Optimizer state:           {theoretical['state']:>8.2f} MB")
        print(f"  {'─'*50}")
        print(f"  Total expected:            {theoretical['total']:>8.2f} MB")
        
        # Compare
        actual_opt_state = deltas['optimizer_state']
        expected_opt_state = theoretical['state']
        print(f"\nOptimizer State Efficiency:")
        print(f"  Measured / Expected:       {actual_opt_state / expected_opt_state:.2%}")
    
    # Clean up
    del test_model
    torch.cuda.empty_cache()
    gc.collect()

# =============================================================================
# Comparison summary
# =============================================================================

print("\n" + "="*80)
print("COMPARATIVE SUMMARY")
print("="*80)

adamw_state = results['AdamW']['optimizer_state']
v5_state = results['BiostatisV5']['optimizer_state']
v51_state = results['BiostatisV5_1']['optimizer_state']
v6_state = results['BiostatisV6']['optimizer_state']

print(f"\nOptimizer State Memory (the key metric):")
print(f"  AdamW:         {adamw_state:>8.2f} MB (baseline)")
print(f"  BiostatisV5:   {v5_state:>8.2f} MB ({v5_state/adamw_state:.2f}× AdamW)")
print(f"  BiostatisV5_1: {v51_state:>8.2f} MB ({v51_state/adamw_state:.2f}× AdamW)")
print(f"  BiostatisV6: {v6_state:>8.2f} MB ({v6_state/adamw_state:.2f}× AdamW)")

print(f"\nTotal Steady State Memory:")
adamw_total = results['AdamW']['total_steady']
v5_total = results['BiostatisV5']['total_steady']
v51_total = results['BiostatisV5_1']['total_steady']
v6_total = results['BiostatisV6']['total_steady']

print(f"  AdamW:         {adamw_total:>8.2f} MB (baseline)")
print(f"  BiostatisV5:   {v5_total:>8.2f} MB ({v5_total/adamw_total:.2f}× AdamW)")
print(f"  BiostatisV5_1: {v51_total:>8.2f} MB ({v51_total/adamw_total:.2f}× AdamW)")
print(f"  BiostatisV6: {v6_total:>8.2f} MB ({v6_total/adamw_total:.2f}× AdamW)")

print(f"\nMemory Overhead Analysis:")
print(f"  BiostatisV5 adds:   {v5_state - adamw_state:.2f} MB extra state")
print(f"  BiostatisV5_1 adds: {v51_state - adamw_state:.2f} MB extra state")
print(f"  BiostatisV6 adds: {v6_state - adamw_state:.2f} MB extra state")

# Theoretical expectation
theoretical_adamw = theoretical_memory(total_params, "AdamW")['state']
theoretical_v5 = theoretical_memory(total_params, "BiostatisV5")['state']
theoretical_v6 = theoretical_memory(total_params, "BiostatisV6")['state']

print(f"\nTheoretical vs Actual Overhead:")
print(f"  Expected ratio (V5/AdamW state): {theoretical_v5/theoretical_adamw:.2f}×")
print(f"  Actual ratio (V5/AdamW state):   {v5_state/adamw_state:.2f}×")
print(f"  Expected ratio (V6/AdamW state): {theoretical_v6/theoretical_adamw:.2f}×")
print(f"  Actual ratio (V6/AdamW state):   {v6_state/adamw_state:.2f}×")

# # Performance vs memory trade-off
# print("\n" + "="*80)
# print("PERFORMANCE vs MEMORY TRADE-OFF")
# print("="*80)
# print(f"\n{'Optimizer':<15} {'Accuracy':<12} {'State Memory':<15} {'vs AdamW':<12}")
# print(f"{'-'*60}")
# print(f"{'AdamW':<15} {'--':<12} {adamw_state:>8.2f} MB    {'1.00×':<12}")
# print(f"{'BiostatisV5':<15} {'50.17%':<12} {v5_state:>8.2f} MB    {v5_state/adamw_state:>5.2f}×")
# print(f"{'BiostatisV5_1':<15} {'52.00%':<12} {v51_state:>8.2f} MB    {v51_state/adamw_state:>5.2f}×")

# print(f"\n{'─'*80}")
# print("Conclusion:")
# print(f"  • BiostatisV5/V5.1 use ~2.5× more optimizer state memory")
# print(f"  • But total training memory is only ~1.4-1.5× (diluted by weights/gradients)")
# print(f"  • V5.1 achieves +2% accuracy over V5 with same memory")
# print(f"  • Trade-off: ~50% more total memory for potentially 10-28% faster convergence")
# print("="*80)