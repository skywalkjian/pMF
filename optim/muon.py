import torch


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int) -> torch.Tensor:
    if grad.ndim < 2:
        raise ValueError(f"Muon expects matrix-like gradients, got shape={tuple(grad.shape)}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.to(torch.bfloat16)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.mT
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if transposed:
        x = x.mT
    return x.to(grad.dtype)


def orthogonalize_update(
    update: torch.Tensor,
    ns_steps: int,
    split_first_dim_chunks: int = 1,
) -> torch.Tensor:
    if update.ndim == 4:
        if split_first_dim_chunks != 1:
            raise ValueError("split_first_dim_chunks is only supported for 2D Muon parameters.")
        update = update.view(len(update), -1)

    if split_first_dim_chunks <= 1:
        orthogonal = zeropower_via_newtonschulz5(update, steps=ns_steps)
        orthogonal *= max(1.0, orthogonal.size(-2) / orthogonal.size(-1)) ** 0.5
        return orthogonal

    if update.ndim != 2:
        raise ValueError("split_first_dim_chunks is only supported for 2D Muon parameters.")
    if update.size(0) % split_first_dim_chunks != 0:
        raise ValueError(
            f"Cannot split shape={tuple(update.shape)} into {split_first_dim_chunks} equal chunks along dim 0."
        )

    chunks = []
    for chunk in update.chunk(split_first_dim_chunks, dim=0):
        orthogonal = zeropower_via_newtonschulz5(chunk, steps=ns_steps)
        orthogonal *= max(1.0, orthogonal.size(-2) / orthogonal.size(-1)) ** 0.5
        chunks.append(orthogonal)
    return torch.cat(chunks, dim=0)


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    split_first_dim_chunks: int = 1,
) -> torch.Tensor:
    # Match the official Muon buffer update so the pre-orthogonalized direction
    # stays aligned with upstream implementations.
    momentum.lerp_(grad, 1.0 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    return orthogonalize_update(update, ns_steps=ns_steps, split_first_dim_chunks=split_first_dim_chunks)


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    exp_avg.lerp_(grad, 1.0 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1.0 - betas[1])
    exp_avg_corr = exp_avg / (1.0 - betas[0] ** step)
    exp_avg_sq_corr = exp_avg_sq / (1.0 - betas[1] ** step)
    return exp_avg_corr / (exp_avg_sq_corr.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Single-device Muon + AdamW wrapper, adapted from the official Muon repo.

    Each param group must include `use_muon=True/False`.
    """

    def __init__(self, param_groups, muon_param_splits: dict[int, int] | None = None):
        normalized_groups = []
        self.muon_param_splits = dict(muon_param_splits or {})
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each param group must define use_muon.")
            params = list(group["params"])
            if not params:
                continue

            if group["use_muon"]:
                normalized_groups.append(
                    {
                        "params": params,
                        "lr": group.get("lr", 0.02),
                        "momentum": group.get("momentum", 0.95),
                        "weight_decay": group.get("weight_decay", 0.0),
                        "ns_steps": group.get("ns_steps", 5),
                        "nesterov": group.get("nesterov", True),
                        "use_muon": True,
                    }
                )
            else:
                normalized_groups.append(
                    {
                        "params": params,
                        "lr": group.get("lr", 3e-4),
                        "betas": group.get("betas", (0.9, 0.95)),
                        "eps": group.get("eps", 1e-10),
                        "weight_decay": group.get("weight_decay", 0.0),
                        "use_muon": False,
                    }
                )

        super().__init__(normalized_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    update = muon_update(
                        param.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                        split_first_dim_chunks=self.muon_param_splits.get(id(param), 1),
                    )
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])
                    param.add_(update.reshape_as(param), alpha=-group["lr"])
            else:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        param.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["lr"])

        return loss
