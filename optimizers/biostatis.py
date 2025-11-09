import torch
import math


class BiostatisV6(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999),
                eps=1e-8, weight_decay=1e-2, homeo_rate=0.03, coherence_target=0.8,
                energy_target=1e-3, lambda_energy=0.1,
                memory_decays=(0.9, 0.99), memory_weights=(0.6, 0.4),
                flip_threshold=0.2, ascent_strength=0.05):
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            homeo_rate=homeo_rate, coherence_target=coherence_target,
            energy_target=energy_target, lambda_energy=lambda_energy,
            memory_decays=memory_decays, memory_weights=memory_weights,
            flip_threshold=flip_threshold, ascent_strength=ascent_strength
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            homeo_rate = group['homeo_rate']
            coherence_target = group['coherence_target']
            energy_target = group['energy_target']
            lambda_energy = group['lambda_energy']
            memory_decays = group['memory_decays']
            memory_weights = group['memory_weights']
            flip_threshold = group['flip_threshold']
            ascent_strength = group['ascent_strength']

            # global statistics (homeostatis sensors)
            grads = [p.grad.view(-1) for p in group['params'] if p.grad is not None]
            if len(grads) == 0:
                continue
            
            g_cat = torch.cat(grads)
            global_energy = torch.mean(g_cat ** 2).item()
            global_coherence = torch.mean(torch.abs(torch.tanh(g_cat))).item()
            
            # Global homeostatic modulation (WEAK influence)
            global_coherence_error = global_coherence - coherence_target
            global_homeo_mod = 1.0 - 0.5 * homeo_rate * torch.tanh(
                torch.tensor(global_coherence_error)
            ).item()  
            
            # Global energy feedback (WEAK influence)
            global_energy_feedback = 1 + 0.5 * lambda_energy * (energy_target - global_energy)
            global_energy_feedback = 0.9 * global_energy_feedback + 0.1
            global_energy_feedback = max(0.925, min(1.075, global_energy_feedback))
            
            # STEP 2: LOCAL ADAPTATION (Per-parameter)
            # ========================================
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['memory_emas'] = [torch.zeros_like(p) for _ in memory_decays]
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                memory_emas = state['memory_emas']
                state['step'] += 1

                # Standard Adam momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Multi-scale memory
                for ema, decay in zip(memory_emas, memory_decays):
                    ema.mul_(decay).add_(grad, alpha=1 - decay)
                
                energy_flow = torch.zeros_like(grad)
                for weight, ema in zip(memory_weights, memory_emas):
                    energy_flow.add_(ema, alpha=weight)

                # Coherence modulation
                flow_change = torch.cosine_similarity(exp_avg.flatten(), grad.flatten(), dim=0)
                polarity = 0.5 * torch.sign(grad) * torch.tanh(flow_change)
                adaptive_grad = grad * (1.0 + polarity)

                # Selective ascent
                importance = exp_avg.abs().mean() / (exp_avg_sq.sqrt().mean() + 1e-12)
                ascent_decay = 1 - math.exp(-0.02 * state['step'])
                if importance < flip_threshold:
                    adaptive_grad.add_(ascent_strength * ascent_decay * grad)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                # LOCAL homeostasis (STRONG influence)
                local_energy = (grad.pow(2).mean()).item()
                local_coherence = torch.mean(torch.abs(torch.tanh(grad))).item()
                
                local_coherence_error = local_coherence - coherence_target
                local_homeo_mod = 1.0 - homeo_rate * torch.tanh(
                    torch.tensor(local_coherence_error)
                ).item()
                
                local_energy_feedback = 1 + lambda_energy * (energy_target - local_energy)
                local_energy_feedback = 0.8 * local_energy_feedback + 0.2
                local_energy_feedback = max(0.85, min(1.15, local_energy_feedback))
                
                # ========================================
                # STEP 3: HIERARCHICAL COMBINATION
                # ========================================
                # Combine global (weak) and local (strong) signals
                # α = 0.3: 30% global, 70% local
                alpha_global = 0.3
                
                combined_homeo_mod = (
                    alpha_global * global_homeo_mod + 
                    (1 - alpha_global) * local_homeo_mod
                )
                
                combined_energy_feedback = (
                    alpha_global * global_energy_feedback + 
                    (1 - alpha_global) * local_energy_feedback
                )
                
                # Final update
                adaptive_update = -step_size * combined_homeo_mod * combined_energy_feedback * (
                    exp_avg / denom + 0.05 * energy_flow + 0.01 * adaptive_grad
                )

                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                p.add_(adaptive_update)

        return loss