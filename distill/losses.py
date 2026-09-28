"""Distillation and Learning-without-Forgetting losses."""

import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, T=4.0):
    """Temperature-scaled KL between the two logit distributions.

    The T^2 factor keeps the gradient magnitude comparable to the hard loss,
    as in Hinton et al. (2015).
    """
    if T <= 0:
        raise ValueError(f"temperature must be positive, got {T}")
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)


def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    """`alpha` soft targets from the teacher, `1 - alpha` hard labels."""
    hard_loss = F.cross_entropy(student_logits, labels)
    if alpha == 0:
        return hard_loss
    return alpha * kd_loss(student_logits, teacher_logits, T) + (1 - alpha) * hard_loss


def lwf_loss(student_logits, old_logits, num_old, T=4.0):
    """Distil the previous snapshot's answers over the *old* classes only."""
    return kd_loss(student_logits[:, :num_old], old_logits[:, :num_old], T)


def icarl_distillation_loss(student_logits, teacher_logits, labels, num_old,
                            replay_ce_weight=0.0):
    """iCaRL / LwF.MC loss: per-node binary cross-entropy over sigmoid outputs, plus an
    optional softmax cross-entropy replay term.

    Old nodes are supervised by the frozen pre-update network's soft targets, new nodes
    by the binary ground-truth label.  Note that an old-class exemplar gets an all-zero
    target on the new nodes — its own label never appears as a hard target, so old-class
    supervision reaches the student only through the teacher.  That is iCaRL as specified.

    On a *frozen* encoder with a single Linear head this specification is degenerate: the
    per-node BCE decouples the output rows, the old rows start as an exact copy of the
    teacher over the exact same features, and their gradient is therefore identically
    zero.  That is not the case here.  The whole CNN trains, so `features(x)` drifts under
    the new task and the old rows' targets stop being met the moment the backbone moves —
    the distillation term is doing real work from the first step rather than none.

    `replay_ce_weight > 0` adds cross-entropy on the true labels.  Its softmax normaliser
    couples every row, so the replay exemplars supervise the old rows directly instead of
    only acting as negatives for the new ones.  This is a deliberate departure from the
    paper — report such runs as replay + LwF, not iCaRL.

    The terms sit on different scales: distill and classification are summed over their
    nodes before the batch mean, while cross-entropy is a single mean, so a weight of 1.0
    already places replay well below distillation.

    Returns (total, distill_term, class_term, replay_term), each as it enters the total,
    so the three parts sum to it.
    """
    targets = torch.zeros_like(student_logits)
    targets[:, :num_old] = torch.sigmoid(teacher_logits[:, :num_old].detach())

    is_new = labels >= num_old
    targets[is_new, labels[is_new]] = 1.0

    per_node = F.binary_cross_entropy_with_logits(student_logits, targets, reduction="none")
    distill = per_node[:, :num_old].sum(dim=1).mean()
    classification = per_node[:, num_old:].sum(dim=1).mean()

    # Applied to the whole batch, not just the exemplars: the new-class images already
    # carry a hard target through `classification`, but keeping cross-entropy over every
    # sample is what makes this arm run the same objective as the cross-entropy baselines
    # it is being compared against.
    if replay_ce_weight:
        replay = replay_ce_weight * F.cross_entropy(student_logits, labels)
    else:
        replay = student_logits.new_zeros(())

    return distill + classification + replay, distill, classification, replay
