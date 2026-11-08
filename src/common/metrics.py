"""Label-based probes (for us, using simulator ground truth) and label-free action-use measures."""
import torch


def ridge_fit(x: torch.Tensor, y: torch.Tensor, alpha: float = 1e-3):
    """Closed-form ridge regression with bias. x: (N, p), y: (N, q). Returns (W, b)."""
    x = x.double()
    y = y.double()
    xm, ym = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    xc, yc = x - xm, y - ym
    p = xc.shape[1]
    scale = (xc.pow(2).sum() / max(1, xc.shape[0])).clamp_min(1e-12)
    a = xc.T @ xc + alpha * scale * xc.shape[0] * torch.eye(p, dtype=x.dtype, device=x.device)
    w = torch.linalg.solve(a, xc.T @ yc)
    b = ym - xm @ w
    return w, b


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean R^2 over output columns."""
    y_true, y_pred = y_true.double(), y_pred.double()
    ss_res = (y_true - y_pred).pow(2).sum(0)
    ss_tot = (y_true - y_true.mean(0, keepdim=True)).pow(2).sum(0).clamp_min(1e-12)
    return float((1 - ss_res / ss_tot).mean())


def probe_r2(x: torch.Tensor, y: torch.Tensor, alpha: float = 1e-3, train_frac: float = 0.5) -> float:
    """Fit a linear probe on the first part, report held-out R^2 on the rest."""
    n = x.shape[0]
    k = int(n * train_frac)
    w, b = ridge_fit(x[:k], y[:k], alpha)
    pred = x[k:].double() @ w + b
    return r2_score(y[k:], pred)


def action_decodability(z_t: torch.Tensor, z_next: torch.Tensor, a_t: torch.Tensor, alpha: float = 1e-3) -> float:
    """Label-free in the sense that it needs only logged actions: R^2 of a_t from (z_t, z_{t+1}).
    Not claimed as new (SLIM 2610.03137, ACPC 2608.12939 and AD-WM 2609.30264 use related probes);
    we use it only to time the predicted escape."""
    return probe_r2(torch.cat([z_t, z_next], dim=1), a_t, alpha)


def counterfactual_action_sensitivity(pred_fn, ctx, a_samples: torch.Tensor, z_next_ref: torch.Tensor) -> float:
    """CAS = mean over states of the variance of the prediction across actions,
    divided by the variance of the true next latent across states.

    pred_fn(ctx, a) -> predicted next latent, (N, d)
    ctx: whatever pred_fn needs (for example a history of latents), batch size N
    a_samples: (K, N, action_dim) alternative actions
    z_next_ref: (N, d) actual next latents, used only for normalisation
    """
    preds = torch.stack([pred_fn(ctx, a_samples[k]) for k in range(a_samples.shape[0])], 0)  # (K, N, d)
    var_a = preds.double().var(0, unbiased=True).sum(-1).mean()
    var_states = z_next_ref.double().var(0, unbiased=True).sum().clamp_min(1e-12)
    return float(var_a / var_states)
