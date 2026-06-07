"""
MetricsTracker class for flexible epoch metrics tracking.

Handles standard metrics (incl. loss, accuracy) and custom metrics with automatic averaging.
"""

import csv
from typing import Any

import torch


class MetricsTracker:
    """
    Flexible metrics tracker for training/evaluation epochs.

    Handles both standard metrics and custom metrics with automatic averaging.
    """

    def __init__(self) -> None:
        """Initialize the MetricsTracker and reset all metric accumulators."""
        self.reset()

    def reset(self) -> None:
        """Reset all metrics for a new epoch."""
        self.total_samples = 0
        self.batch_count = 0
        self.num_tokens_labels_ag_to_ab = 0
        self.num_tokens_labels_ab_to_ag = 0

        # self.total_loss = 0.0
        self.total_loss_ag_to_ab = 0.0
        self.total_loss_ab_to_ag = 0.0

        self.total_ppl = 0.0
        self.total_ppl_ag_to_ab = 0.0
        self.total_ppl_ab_to_ag = 0.0

        self.sum_token_acc_per_seq_ag_to_ab = 0.0
        self.sum_token_acc_per_seq_ab_to_ag = 0.0
        self.avg_token_acc_per_seq_ag_to_ab = 0.0
        self.avg_token_acc_per_seq_ab_to_ag = 0.0

        self.total_grad_norm = 0.0
        self.max_grad_norm = 0.0

        # Additional metrics storage
        self.custom_metrics: dict[str, list[float]] = {}

    def update_batch(
        self,
        batch: dict[str, Any],
        ag_to_ab_output: Any,
        ab_to_ag_output: Any,
        grad_norm: float,
    ) -> None:
        """Update metrics for current batch.

        Parameters
        ----------
        batch : dict[str, Any]
            Input batch containing 'ag2ab' and 'ab2ag' sub-dicts, each with
            'labels' and 'logits' tensors.
        ag_to_ab_output : Any
            Output from the ag→ab model pass containing 'loss' and 'logits'.
        ab_to_ag_output : Any
            Output from the ab→ag model pass containing 'loss' and 'logits'.
        grad_norm : float
            Gradient norm for the current batch.
        """
        labels_ag_to_ab = batch["ag2ab"]["labels"].detach().cpu()
        labels_ab_to_ag = batch["ab2ag"]["labels"].detach().cpu()
        loss_ag_to_ab = ag_to_ab_output["loss"].item()
        loss_ab_to_ag = ab_to_ag_output["loss"].item()
        logits_ag_to_ab = (
            ag_to_ab_output["logits"].detach().cpu()
        )  # (batch_size, seq_len_dec, vocab_size)
        logits_ab_to_ag = ab_to_ag_output["logits"].detach().cpu()

        batch_size = labels_ag_to_ab.size(0)
        self.total_samples += batch_size
        self.batch_count += 1

        # token counts
        num_tokens_labels_ag_to_ab = (labels_ag_to_ab != -100).sum().item()
        num_tokens_labels_ab_to_ag = (labels_ab_to_ag != -100).sum().item()

        self.num_tokens_labels_ag_to_ab += num_tokens_labels_ag_to_ab
        self.num_tokens_labels_ab_to_ag += num_tokens_labels_ab_to_ag

        self.total_loss_ag_to_ab += loss_ag_to_ab * num_tokens_labels_ag_to_ab
        self.total_loss_ab_to_ag += loss_ab_to_ag * num_tokens_labels_ab_to_ag

        self.loss_ag_to_ab = (
            self.total_loss_ag_to_ab / self.num_tokens_labels_ag_to_ab
            if self.num_tokens_labels_ag_to_ab > 0
            else 0.0
        )
        self.loss_ab_to_ag = (
            self.total_loss_ab_to_ag / self.num_tokens_labels_ab_to_ag
            if self.num_tokens_labels_ab_to_ag > 0
            else 0.0
        )
        self.loss = (self.total_loss_ag_to_ab + self.total_loss_ab_to_ag) / (
            self.num_tokens_labels_ag_to_ab + self.num_tokens_labels_ab_to_ag
        )

        self.ppl_ag_to_ab = torch.exp(torch.tensor(self.loss_ag_to_ab)).item()
        self.ppl_ab_to_ag = torch.exp(torch.tensor(self.loss_ab_to_ag)).item()
        self.ppl = torch.exp(torch.tensor(self.loss)).item()

        # token-level accuracy
        preds_ag_to_ab = logits_ag_to_ab.argmax(dim=2)  # (batch_size, seq_len_dec)
        preds_ab_to_ag = logits_ab_to_ag.argmax(dim=2)
        mask_ag_to_ab = labels_ag_to_ab != -100  # (batch_size, seq_len_dec)
        mask_ab_to_ag = labels_ab_to_ag != -100

        correct_ag_to_ab = (preds_ag_to_ab == labels_ag_to_ab) & mask_ag_to_ab
        correct_ab_to_ag = (preds_ab_to_ag == labels_ab_to_ag) & mask_ab_to_ag
        token_acc_per_seq_ag_to_ab = correct_ag_to_ab.sum(dim=1) / mask_ag_to_ab.sum(
            dim=1
        ).clamp_min(1)
        token_acc_per_seq_ab_to_ag = correct_ab_to_ag.sum(dim=1) / mask_ab_to_ag.sum(
            dim=1
        ).clamp_min(1)

        self.sum_token_acc_per_seq_ag_to_ab += sum(token_acc_per_seq_ag_to_ab).item()
        self.sum_token_acc_per_seq_ab_to_ag += sum(token_acc_per_seq_ab_to_ag).item()

        self.avg_token_acc_per_seq_ag_to_ab = (
            self.sum_token_acc_per_seq_ag_to_ab / self.total_samples
        )
        self.avg_token_acc_per_seq_ab_to_ag = (
            self.sum_token_acc_per_seq_ab_to_ag / self.total_samples
        )

        self.total_grad_norm += grad_norm
        self.max_grad_norm = max(self.max_grad_norm, grad_norm)

    def add_custom_metric(self, name: str, value: float) -> None:
        """Add a custom metric value for the current epoch.

        Parameters
        ----------
        name : str
            Name of the metric.
        value : float
            Value to record.
        """
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        self.custom_metrics[name].append(value)

    def get_epoch_results(self) -> dict[str, float]:
        """Calculate and return final epoch metrics.

        Returns
        -------
        dict[str, float]
            Dictionary containing all calculated metrics.
        """
        if self.total_samples == 0:
            return self._get_empty_results()

        results = {
            "samples": self.total_samples,
            "batch_count": self.batch_count,
            "num_tokens_labels_ag_to_ab": self.num_tokens_labels_ag_to_ab,
            "num_tokens_labels_ab_to_ag": self.num_tokens_labels_ab_to_ag,
            # Core metrics
            "loss": self.loss,
            "loss_ag_to_ab": (self.loss_ag_to_ab),
            "loss_ab_to_ag": (self.loss_ab_to_ag),
            "ppl": self.ppl,
            "ppl_ag_to_ab": (self.ppl_ag_to_ab),
            "ppl_ab_to_ag": (self.ppl_ab_to_ag),
            # Token-level accuracy metrics
            "avg_token_acc_per_seq_ag_to_ab": self.avg_token_acc_per_seq_ag_to_ab,
            "avg_token_acc_per_seq_ab_to_ag": self.avg_token_acc_per_seq_ab_to_ag,
            "avg_token_acc": (
                self.avg_token_acc_per_seq_ag_to_ab
                + self.avg_token_acc_per_seq_ab_to_ag
            )
            / 2,
            # Derived metrics
            "avg_grad_norm": (
                self.total_grad_norm / self.batch_count if self.batch_count > 0 else 0.0
            ),
            "max_grad_norm": self.max_grad_norm,
        }

        # Add custom metrics (averaged)
        for metric_name, values in self.custom_metrics.items():
            if values:
                results[f"avg_{metric_name}"] = sum(values) / len(values)
                results[f"total_{metric_name}"] = sum(values)

        return results

    def _get_empty_results(self) -> dict[str, float]:
        """Return empty results when no samples processed.

        Returns
        -------
        dict
            Dictionary containing empty metric values
        """
        return {
            "samples": 0,
            "batch_count": 0,
            "num_tokens_labels_ag_to_ab": 0,
            "num_tokens_labels_ab_to_ag": 0,
            # Core metrics
            "loss": 0.0,
            "loss_ag_to_ab": 0.0,
            "loss_ab_to_ag": 0.0,
            "ppl": 0.0,
            "ppl_ag_to_ab": 0.0,
            "ppl_ab_to_ag": 0.0,
            # Token-level accuracy metrics
            "avg_token_acc_per_seq_ag_to_ab": 0.0,
            "avg_token_acc_per_seq_ab_to_ag": 0.0,
            "avg_token_acc": 0.0,
            # Derived metrics
            "avg_grad_norm": 0.0,
            "max_grad_norm": 0.0,
        }

    def get_summary_string(self, phase: str) -> str:
        """Get a formatted summary string of current metrics.

        Parameters
        ----------
        phase : str
            Phase name for formatting.

        Returns
        -------
        str
            Formatted summary string.
        """
        results = self.get_epoch_results()
        phase_str = f"{phase} " if phase else ""

        return (
            f"{phase_str}Loss: {results['loss']:.4f}, "
            f"Acc_AG: {results['avg_token_acc_per_seq_ag_to_ab']:.4f}, "
            f"Acc_AB: {results['avg_token_acc_per_seq_ab_to_ag']:.4f}, "
            f"Samples: {results['samples']}"
        )


def save_results(
    metrics_dict: dict[str, list[dict[str, float]]], file_out: str
) -> None:
    """Save epoch metrics to a CSV file.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary with metric dicts per phase per epoch
            - "train": [ {metric_dict_epoch1}, {metric_dict_epoch2}, ... ]
            - "val": [ {metric_dict_epoch1}, {metric_dict_epoch2}, ... ]
            - "test": [ {metric_dict_epoch1}, {metric_dict_epoch2}, ...
    file_out : str
        Output CSV filename
    """
    rows = []
    for phase, metrics_list in metrics_dict.items():
        for entry in metrics_list:
            row = {"phase": phase, **entry}
            rows.append(row)

    if not rows:
        print("No metrics to save.")
        return

    header = sorted(rows[0].keys())
    try:
        with open(file_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Metrics saved to {file_out}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
