"""BiC stage 2: fit a small calibration on a balanced held-out set, then fold it in.

Three modes, all of which leave the network an ordinary classifier once folded, so
nothing downstream (evaluate, the checkpoint, the next task) needs to know a correction
was applied:

  scalar        q_k = alpha*o_k + beta for the new classes only   (Wu et al., BiC)
  vector        q   = o + b,  one additive offset per class
  vector_alpha  q   = alpha*o + b,  those offsets plus a shared temperature
"""

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

MODES = ("scalar", "vector", "vector_alpha")


@torch.no_grad()
def _cache_logits(model, loader, device):
    """(logits, labels) per batch, computed once.

    Stage 2 freezes the network, so the logits cannot change while the calibration
    parameters are fit — re-running the CNN every epoch would cost orders of magnitude
    more than the fit itself.
    """
    model.eval()
    return [(model(images.to(device)), labels.to(device)) for images, labels in loader]


def _warn_if_unconverged(label, initial, late, final):
    """Flag a calibration that stopped because it ran out of epochs, not because it converged.

    Adam moves each parameter by at most ~lr per step, so the whole fit can travel no
    further than epochs x batches x lr.  That is ample when the head is nearly calibrated
    already and nowhere near enough when it is badly skewed, and the two cases look
    identical in the final numbers.  A converged fit's parameters are barely moving by the
    end; one still travelling in its last tenth is reporting the budget rather than the data.
    """
    total = float((final - initial).norm())
    tail = float((final - late).norm())
    # A steady-rate fit puts a full tenth of its travel in its last tenth; 5% separates them.
    if total > 0 and tail > 0.05 * total:
        log.warning("%s was still moving over the last 10%% of the fit (%.0f%% of its total "
                    "travel) - it is bounded by the step budget, not converged.  Raise "
                    "--bic-epochs or --bic-lr.", label, 100 * tail / total)


def fit_bias_correction(model, loader, num_old, device, epochs=200, lr=1e-3):
    """BiC stage 2: fit the two bias parameters on a small balanced validation set.

    Corrects only the new-class logits (q_k = alpha*o_k + beta for k >= num_old) under
    softmax cross-entropy, with the rest of the network frozen.

    Weight decay is off: AdamW's default would pull alpha and beta towards 0 rather than
    towards the identity correction (alpha=1, beta=0), which on a few hundred held-out
    images is a large and entirely unintended shrinkage.
    """
    cached = _cache_logits(model, loader, device)
    if not cached:
        log.warning("bias-correction set was empty - leaving logits uncorrected (a=1, b=0).")
        return 1.0, 0.0

    alpha = torch.ones(1, device=device, requires_grad=True)
    beta = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([alpha, beta], lr=lr, weight_decay=0.0)

    initial = torch.tensor([1.0, 0.0])
    late = initial.clone()

    for epoch in range(epochs):
        if epoch == int(0.9 * epochs):
            late = torch.tensor([alpha.item(), beta.item()])
        total = 0.0
        for logits, labels in cached:
            corrected = torch.cat(
                [logits[:, :num_old], alpha * logits[:, num_old:] + beta], dim=1)
            loss = F.cross_entropy(corrected, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        if (epoch + 1) % 50 == 0:
            log.info("  bias-correction epoch %3d/%d | loss %.4f | a=%.4f b=%.4f",
                     epoch + 1, epochs, total / len(cached), alpha.item(), beta.item())

    alpha, beta = float(alpha.item()), float(beta.item())
    _warn_if_unconverged("alpha/beta", initial, late, torch.tensor([alpha, beta]))
    if alpha <= 0:
        # Nothing in Eq. 4/5 constrains alpha to be positive, and on a very small
        # validation set it can fit to a sign flip, inverting the new-class ranking.
        log.warning("bias correction fitted a=%.4f (<= 0) - the validation set is likely "
                    "too small; consider raising --bic-val-per-class.", alpha)
    return alpha, beta


@torch.no_grad()
def fold_bias_correction(model, alpha, beta, num_old):
    """Absorb (alpha, beta) into the new-class rows of the output layer.

    a*(w_k.f + b_k) + b == (a*w_k).f + (a*b_k + b), so the corrected network is an
    ordinary classifier and nothing downstream (evaluate.py, save_checkpoint, the next
    task) needs to know a correction was applied.
    """
    out = model.output_layer
    out.weight[num_old:] *= alpha
    out.bias[num_old:] = out.bias[num_old:] * alpha + beta


def fit_bias_vector(model, loader, num_classes, device, epochs=200, lr=1e-3,
                    fit_alpha=False):
    """Bias-only vector scaling: one additive offset per class, fit on the same balanced set.

    q = alpha*o + b with b in R^num_classes, under softmax cross-entropy and with the rest
    of the network frozen.  alpha is pinned at 1 unless fit_alpha is set - that is the
    bias-only case.  Where fit_bias_correction rescales only the new-class logits and
    leaves the old ones untouched, this gives every class - old and new - its own offset,
    so the correction can also push down an old class the new ones are being confused
    with.  It buys that with strictly less power in another direction: no *class* gets its
    own scale, so the decision boundaries shift but never change orientation.

    What the shared alpha does, and does not, do.  For alpha > 0,
    argmax(a*o + b) == argmax(o + b/a), so it adds no decision rule the bias-only fit could
    not already express: the two are the same family, reparameterised.  It earns its place
    anyway, for two reasons.  It is a temperature, so it rescales every confidence the
    model reports - and sigmoid(teacher_logits) at the next task, where absolute logit
    level does matter.  And under a fixed step budget it changes what is *reachable*:
    alpha and b travel together, and the effective offset b/a can end up larger than
    anything b reaches alone in the same number of steps.  (Measured on a synthetic
    4-class head in the voice-command study this was ported from, the new class's
    effective offset came out -0.093 with alpha fit against -0.068 without.)  See
    _warn_if_unconverged for why that ceiling binds at all.

    alpha <= 0 inverts the ranking of every class at once, so it is warned about the same
    way fit_bias_correction warns about its own.

    Softmax is invariant to a constant added to every logit, so b is only identified up to
    that constant: the cross-entropy gradient w.r.t. b sums to zero across classes, and
    plain gradient descent from b=0 would therefore stay zero-sum.  Adam's per-parameter
    normalisation breaks that, letting a meaningless common offset accumulate.  The
    returned vector is re-centred to sum to zero - a no-op for this model's predictions,
    but it keeps the logged offsets comparable across runs, and stops an arbitrary shift
    from leaking into sigmoid(teacher_logits) when this model becomes the teacher for the
    next task, where absolute logit level does matter.

    Weight decay is off for the same reason as in fit_bias_correction: b=0 is the identity
    correction, and shrinking towards it on a small held-out set is not a prior anyone chose.
    """
    cached = _cache_logits(model, loader, device)

    if not cached:
        log.warning("bias-correction set was empty - leaving logits uncorrected (a=1, b=0).")
        return 1.0, [0.0] * num_classes

    n_logits = cached[0][0].shape[1]
    if n_logits != num_classes:
        raise ValueError(f"num_classes={num_classes} but the head emits {n_logits} logits")

    bias = torch.zeros(num_classes, device=device, requires_grad=True)
    alpha = torch.ones(1, device=device, requires_grad=fit_alpha)
    optimizer = torch.optim.AdamW([bias, alpha] if fit_alpha else [bias],
                                  lr=lr, weight_decay=0.0)

    # alpha and b are fit jointly, so convergence has to be judged on both together.
    def trajectory():
        parts = [bias.detach().cpu()] + ([alpha.detach().cpu()] if fit_alpha else [])
        return torch.cat(parts)

    initial = torch.cat([torch.zeros(num_classes)] + ([torch.ones(1)] if fit_alpha else []))
    late = initial.clone()

    for epoch in range(epochs):
        if epoch == int(0.9 * epochs):
            late = trajectory()
        total = 0.0
        for logits, labels in cached:
            loss = F.cross_entropy(alpha * logits + bias, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        if (epoch + 1) % 50 == 0:
            # Centred, so these lines agree with the vector this eventually returns.
            shown = bias.detach() - bias.detach().mean()
            log.info("  bias-vector epoch %3d/%d | loss %.4f | a=%.4f | ||b||=%.4f",
                     epoch + 1, epochs, total / len(cached), alpha.item(),
                     float(shown.norm()))

    _warn_if_unconverged("bias vector", initial, late, trajectory())
    # Re-centring is a no-op for softmax, so it is applied after the convergence check
    # rather than before - the check should see the distance the fit actually travelled.
    centred = (bias.detach().cpu() - bias.detach().mean().cpu()).tolist()
    alpha = float(alpha.item())
    if fit_alpha and alpha <= 0:
        log.warning("bias vector fitted a=%.4f (<= 0) - this inverts the ranking of every "
                    "class at once; the fitting set is likely too small.", alpha)
    return alpha, [float(v) for v in centred]


@torch.no_grad()
def fold_bias_vector(model, bias_vector, alpha=1.0):
    """Absorb the per-class offsets, and any shared alpha, into the output layer.

    a*(w_k.f + b_k) + v_k == (a*w_k).f + (a*b_k + v_k), so - exactly as with
    fold_bias_correction - the corrected network stays an ordinary classifier and nothing
    downstream needs to know a correction was applied.  Unlike fold_bias_correction, alpha
    here scales *every* row, old classes included: it is a temperature on the whole head,
    not a reweighting of the new classes against the old ones.

    alpha = 1.0, the bias-only default, leaves the weights untouched and reduces this to
    out.bias += v.
    """
    out = model.output_layer
    v = torch.as_tensor(bias_vector, dtype=out.bias.dtype, device=out.bias.device)
    if v.numel() != out.bias.numel():
        raise ValueError(f"bias vector has {v.numel()} entries but the head has "
                         f"{out.bias.numel()} classes")
    if alpha != 1.0:
        out.weight *= alpha
    out.bias.copy_(out.bias * alpha + v)


def apply_bias_correction(model, bic_loader, num_old, num_seen, device, mode="scalar",
                          epochs=200, lr=1e-3):
    """Fit the chosen calibration, fold it into the head, and return a record of it."""
    if mode not in MODES:
        raise ValueError(f"unknown bias-correction mode {mode!r}; choose from {list(MODES)}")

    if mode == "scalar":
        alpha, beta = fit_bias_correction(model, bic_loader, num_old, device, epochs, lr)
        fold_bias_correction(model, alpha, beta, num_old)
        log.info("  a=%.4f  b=%.4f", alpha, beta)
        return {"mode": mode, "alpha": alpha, "beta": beta,
                "num_old": num_old, "folded": True}

    fit_alpha = mode == "vector_alpha"
    alpha, bias_vector = fit_bias_vector(model, bic_loader, num_seen, device,
                                         epochs, lr, fit_alpha=fit_alpha)
    fold_bias_vector(model, bias_vector, alpha)
    # alpha is recorded either way - it is 1.0 in bias-only mode, which makes the stored
    # transform readable without having to know which mode wrote it.
    new_mean = sum(bias_vector[num_old:]) / max(num_seen - num_old, 1)
    old_mean = sum(bias_vector[:num_old]) / num_old if num_old else 0.0
    log.info("  %sb over %d classes: new-class mean %+.4f, old-class mean %+.4f",
             f"a={alpha:.4f}  " if fit_alpha else "", num_seen, new_mean, old_mean)
    return {"mode": mode, "alpha": alpha, "bias": bias_vector,
            "num_old": num_old, "folded": True}
