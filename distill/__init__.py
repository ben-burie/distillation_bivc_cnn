"""Class-incremental learning: distillation, replay and bias correction.

The pipeline is split into single-purpose modules:

  config           command line parsing and validation
  utils            device selection, seeding, logging, checkpoint helpers
  data             dataset construction and the class-subset plumbing
  splits           the train / bic / val partition for one task
  models           the CNN backbones and the growable head
  losses           KD, iCaRL/LwF.MC and softmax LwF losses
  bias_correction  the stage-2 calibrations, and folding them into the head
  train            the training loop
  evaluate         accuracy, the old/new breakdown, per-task evaluation
  memory           the replay exemplar memory
  metrics          accuracy bookkeeping and the end-of-run summary
  experiments      the two top-level experiments (joint, incremental)
"""

__all__ = ["bias_correction", "config", "data", "evaluate", "losses", "memory",
           "metrics", "models", "splits", "train", "utils"]
