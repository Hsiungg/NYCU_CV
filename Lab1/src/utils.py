"""Utility functions"""
import random
import numpy as np
import evaluate
import torch

accuracy_metric = evaluate.load("accuracy")


def count_parameters(model):
    """Returns the number of parameters in pytorch model"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


def compute_metrics(eval_pred):
    """set up metrix use in validation """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    if labels.ndim > 1:  # add soft labels for mixup strategies
        labels = np.argmax(labels, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"eval_accuracy": acc["accuracy"]}


def same_seeds(seed):
    """Fix the seed """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
