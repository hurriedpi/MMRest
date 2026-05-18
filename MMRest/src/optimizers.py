import math
from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class PSDAdam(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PSDAdam does not support sparse gradients")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Parameter update
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # PSD projection for matrix-shaped parameters
                if p.ndim >= 2 and p.shape[-1] == p.shape[-2]:
                    # Symmetrize and upcast for numerical stability
                    orig_dtype = p.data.dtype
                    M = 0.5 * (p.data + p.data.transpose(-1, -2))
                    M = M.to(torch.float64)

                    def project_psd_matrix(Mi: torch.Tensor) -> torch.Tensor:
                        # Try eigen approach with diagonal jitter
                        eye = torch.eye(Mi.shape[-1], dtype=Mi.dtype, device=Mi.device)
                        jitter = 0.0
                        for _ in range(6):
                            try:
                                M_try = Mi if jitter == 0.0 else (Mi + jitter * eye)
                                evals, evecs = torch.linalg.eigh(M_try)
                                evals = torch.clamp(evals, min=0.0)
                                return evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
                            except RuntimeError:
                                jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
                        # Fallback: Cholesky with added jitter if possible
                        jitter = 1e-8
                        for _ in range(6):
                            L, info = torch.linalg.cholesky_ex(Mi + jitter * eye)
                            if int(info.item()) == 0:
                                return L @ L.transpose(-1, -2)
                            jitter *= 10.0
                        # Last resort: clamp diagonal to small positive
                        M_diag = torch.diagonal(Mi, dim1=-2, dim2=-1)
                        M_diag.clamp_(min=1e-12)
                        return Mi

                    try:
                        if M.ndim > 2:
                            M_flat = M.reshape(-1, M.shape[-2], M.shape[-1])
                            outs = [project_psd_matrix(M_flat[i]) for i in range(M_flat.shape[0])]
                            M_psd = torch.stack(outs, dim=0).reshape_as(M)
                        else:
                            M_psd = project_psd_matrix(M)
                        p.data.copy_(M_psd.to(orig_dtype))
                    except Exception:
                        # Ensure at least symmetry is preserved
                        p.data.copy_(M.to(orig_dtype))

        return loss
