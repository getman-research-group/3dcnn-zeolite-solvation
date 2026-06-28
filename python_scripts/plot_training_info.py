# -*- coding: utf-8 -*-
"""Generate the publication-ready training-history figure for the 3D-CNN.

This script reads the cross-validation results pickle produced by
``train_3d_cnn.py``. For every available fold, it extracts the stored training
loss, validation loss, gradient-norm history, and target-scaler information.
The result pickle is loaded with a CPU-safe unpickler so that files containing
PyTorch tensors can be analyzed on a machine without the original GPU device.

The ``plot_training_info_publish`` method generates a two-panel figure:

1. Mean training and validation loss across folds, with shaded bands showing
   one standard deviation between folds. Stored MSE values can be displayed
   directly or converted to RMSE; fold-specific target scalers can optionally
   restore the original energy unit (eV).
2. Mean gradient norm across folds, again with a one-standard-deviation band,
   to summarize the optimization behavior during training.

The plotting interface provides optional epoch trimming, validation-curve
smoothing, axis limits, best-epoch marking, and standardized/original-scale
display controls. Running this file directly loads ``output_model_cnn/model.pkl``
and writes the figure to ``output_figures/cnn_training_results/``.

Main components
---------------
``EnhancedTrainingPlotter``
    Resolves repository paths, loads the stored fold histories, and manages
    figure output settings.
``load_model_data``
    Reads the result pickle and retains only the fields required by the
    publication plot.
``plot_training_info_publish``
    Aggregates the histories across folds and generates the final two-panel
    training-information figure.
"""

import io
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from core.path import get_paths


class EnhancedTrainingPlotter:
    """Load training histories and generate the publication figure."""

    def __init__(
        self,
        model_file=None,
        output_dir=None,
        verbose=True,
        show_plot=False,
        save_plot=True,
        font_size=18,
    ):
        self.verbose = verbose
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.font_size = font_size
        self.model_dir = get_paths("output_model_cnn")
        self.output_dir = output_dir or os.path.join(
            get_paths("output_figure_path"), "cnn_training_results"
        )
        self.model_file = model_file
        self.model_name = os.path.basename(model_file) if model_file else None
        self.training_data = {"folds": {}}

        os.makedirs(self.output_dir, exist_ok=True)

        if self.verbose:
            print("=== Training Information Plotter Initialized ===")
            print(f"Model directory: {self.model_dir}")
            print(f"Output directory: {self.output_dir}")

        if self.model_file:
            self.load_model_data(self.model_file)

    def load_model_data(self, model_file):
        """Load fold histories from a result pickle using CPU-safe unpickling."""
        if os.path.isabs(model_file):
            file_path = model_file
        else:
            candidates = [
                os.path.join(self.model_dir, model_file),
                model_file,
                os.path.join(os.getcwd(), model_file),
            ]
            file_path = next((path for path in candidates if os.path.exists(path)), None)

        if file_path is None:
            raise FileNotFoundError(f"Model result file not found: {model_file}")

        if self.verbose:
            print(f"Loading model file: {file_path}")

        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda data: torch.load(io.BytesIO(data), map_location="cpu")
                return super().find_class(module, name)

        with open(file_path, "rb") as file:
            data = CPUUnpickler(file).load()

        if "model_storage" not in data:
            raise KeyError("The result pickle does not contain 'model_storage'.")

        folds = {}
        for fold_idx, fold_data in data["model_storage"].items():
            monitoring = fold_data.get("monitoring_data") or {}
            folds[fold_idx] = {
                "train_losses": monitoring.get("train_losses", []),
                "test_losses": monitoring.get("test_losses", []),
                "gradient_norms": monitoring.get("gradient_norms", []),
                "scaler_info": fold_data.get("scaler_info"),
            }

        self.training_data = {"folds": folds}
        self.model_name = os.path.basename(file_path)

        if self.verbose:
            folds_with_curves = sum(bool(fold["train_losses"]) for fold in folds.values())
            folds_with_gradients = sum(bool(fold["gradient_norms"]) for fold in folds.values())
            print(f"Successfully loaded {len(folds)} folds")
            print(f"Folds with loss histories: {folds_with_curves}")
            print(f"Folds with gradient histories: {folds_with_gradients}")

        return True

    def plot_training_info_publish(
        self,
        x_pos=-0.16,
        y_pos=1.08,
        loss_y_max=None,
        loss_y_min=None,
        train_loss_start_epoch=1,
        val_loss_start_epoch=1,
        gradient_y=None,
        gradient_start_epoch=1,
        show_best_epoch=False,
        convert_to_original_scale=True,
        plot_raw_training_loss=False,
        smooth_val_loss=False,
        smooth_window=5,
    ):
        """Create the two-panel publication training-history figure.

        The left panel reports fold-averaged training and validation histories.
        By default, stored standardized MSE values are converted to RMSE in eV
        with each fold's target scaler. The right panel reports fold-averaged
        gradient norms. Shaded regions represent ±1 standard deviation.
        """
        folds = self.training_data["folds"]
        folds_with_curves = [
            idx
            for idx, fold in folds.items()
            if fold.get("train_losses") and fold.get("test_losses")
        ]
        folds_with_gradients = [
            idx for idx, fold in folds.items() if fold.get("gradient_norms")
        ]

        if not folds_with_curves:
            raise ValueError("No complete training and validation loss histories are available.")
        if not folds_with_gradients:
            raise ValueError("No gradient-norm histories are available.")

        fold_scalers = {
            idx: folds[idx].get("scaler_info")
            for idx in folds_with_curves
            if folds[idx].get("scaler_info")
        }
        if convert_to_original_scale and len(fold_scalers) != len(folds_with_curves):
            print("Fold-specific scalers are incomplete; displaying standardized RMSE.")
            convert_to_original_scale = False

        def convert_loss(mse_loss, fold_idx):
            if plot_raw_training_loss:
                return mse_loss
            rmse = np.sqrt(mse_loss)
            if convert_to_original_scale:
                return rmse * fold_scalers[fold_idx]["std"]
            return rmse

        def smooth_curve(values, window_size):
            if window_size < 2 or len(values) < window_size:
                return np.asarray(values)
            values = np.asarray(values)
            smoothed = []
            half_window = window_size // 2
            for idx in range(len(values)):
                start = max(0, idx - half_window)
                stop = min(len(values), idx + half_window + 1)
                smoothed.append(np.mean(values[start:stop]))
            return np.asarray(smoothed)

        max_epochs = max(len(folds[idx]["train_losses"]) for idx in folds_with_curves)
        avg_train_losses = []
        std_train_losses = []
        avg_val_losses = []
        std_val_losses = []

        for epoch in range(max_epochs):
            train_values = [
                convert_loss(folds[idx]["train_losses"][epoch], idx)
                for idx in folds_with_curves
                if epoch < len(folds[idx]["train_losses"])
            ]
            val_values = [
                convert_loss(folds[idx]["test_losses"][epoch], idx)
                for idx in folds_with_curves
                if epoch < len(folds[idx]["test_losses"])
            ]
            if train_values:
                avg_train_losses.append(np.mean(train_values))
                std_train_losses.append(np.std(train_values))
            if val_values:
                avg_val_losses.append(np.mean(val_values))
                std_val_losses.append(np.std(val_values))

        train_epochs = np.arange(1, len(avg_train_losses) + 1)
        val_epochs = np.arange(1, len(avg_val_losses) + 1)
        avg_train_losses = np.asarray(avg_train_losses)
        std_train_losses = np.asarray(std_train_losses)
        avg_val_losses = np.asarray(avg_val_losses)
        std_val_losses = np.asarray(std_val_losses)

        train_start = max(0, train_loss_start_epoch - 1)
        val_start = max(0, val_loss_start_epoch - 1)
        train_epochs = train_epochs[train_start:]
        avg_train_losses = avg_train_losses[train_start:]
        std_train_losses = std_train_losses[train_start:]

        if smooth_val_loss:
            avg_val_losses = smooth_curve(avg_val_losses, smooth_window)
            std_val_losses = smooth_curve(std_val_losses, max(3, smooth_window // 2))
        val_epochs = val_epochs[val_start:]
        avg_val_losses = avg_val_losses[val_start:]
        std_val_losses = std_val_losses[val_start:]

        plt.style.use("default")
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        ax_loss = axes[0]
        ax_loss.plot(
            train_epochs,
            avg_train_losses,
            color=colors[0],
            linewidth=2.5,
            label="Training Loss",
        )
        ax_loss.fill_between(
            train_epochs,
            avg_train_losses - std_train_losses,
            avg_train_losses + std_train_losses,
            color=colors[0],
            alpha=0.25,
        )
        ax_loss.plot(
            val_epochs,
            avg_val_losses,
            color=colors[1],
            linewidth=2.5,
            label="Validation Loss",
        )
        ax_loss.fill_between(
            val_epochs,
            avg_val_losses - std_val_losses,
            avg_val_losses + std_val_losses,
            color=colors[1],
            alpha=0.25,
        )

        if show_best_epoch:
            best_idx = int(np.argmin(avg_val_losses))
            best_epoch = val_epochs[best_idx]
            best_loss = avg_val_losses[best_idx]
            ax_loss.axvline(best_epoch, color="red", linestyle="--", alpha=0.7)
            ax_loss.plot(best_epoch, best_loss, "ro", markersize=8)

        ax_loss.set_xlabel("Number of epochs", fontsize=self.font_size)
        if plot_raw_training_loss:
            ax_loss.set_ylabel("MSE Loss (standardized)", fontsize=self.font_size)
        elif convert_to_original_scale:
            ax_loss.set_ylabel("RMSE (eV)", fontsize=self.font_size)
        else:
            ax_loss.set_ylabel("RMSE (standardized)", fontsize=self.font_size)
        ax_loss.text(
            x_pos,
            y_pos,
            "(a)",
            transform=ax_loss.transAxes,
            fontsize=self.font_size,
            fontweight="bold",
            va="top",
        )
        ax_loss.legend(loc="upper right", fontsize=self.font_size)
        ax_loss.tick_params(labelsize=self.font_size)

        auto_y_min = 0.9 * min(np.min(avg_train_losses), np.min(avg_val_losses))
        y_min = loss_y_min if loss_y_min is not None else max(0, auto_y_min)
        if loss_y_max is None:
            ax_loss.set_ylim(bottom=y_min)
        else:
            ax_loss.set_ylim(bottom=y_min, top=loss_y_max)

        max_grad_epochs = max(
            len(folds[idx]["gradient_norms"]) for idx in folds_with_gradients
        )
        avg_grad_norms = []
        std_grad_norms = []
        for epoch in range(max_grad_epochs):
            values = [
                folds[idx]["gradient_norms"][epoch]
                for idx in folds_with_gradients
                if epoch < len(folds[idx]["gradient_norms"])
            ]
            if values:
                avg_grad_norms.append(np.mean(values))
                std_grad_norms.append(np.std(values))

        grad_start = max(0, gradient_start_epoch - 1)
        grad_epochs = np.arange(1, len(avg_grad_norms) + 1)[grad_start:]
        avg_grad_norms = np.asarray(avg_grad_norms)[grad_start:]
        std_grad_norms = np.asarray(std_grad_norms)[grad_start:]

        ax_gradient = axes[1]
        ax_gradient.plot(
            grad_epochs,
            avg_grad_norms,
            color=colors[2],
            linewidth=3,
            label="Average Gradient Norm",
        )
        ax_gradient.fill_between(
            grad_epochs,
            avg_grad_norms - std_grad_norms,
            avg_grad_norms + std_grad_norms,
            color=colors[2],
            alpha=0.25,
            label="±1σ across folds",
        )
        ax_gradient.set_xlabel("Number of epochs", fontsize=self.font_size)
        ax_gradient.set_ylabel("Gradient Norm", fontsize=self.font_size)
        ax_gradient.text(
            x_pos,
            y_pos,
            "(b)",
            transform=ax_gradient.transAxes,
            fontsize=self.font_size,
            fontweight="bold",
            va="top",
        )
        ax_gradient.legend(loc="best", fontsize=self.font_size)
        ax_gradient.tick_params(labelsize=self.font_size)

        if gradient_y is not None:
            ax_gradient.set_ylim(top=gradient_y)
        positive_gradients = avg_grad_norms[avg_grad_norms > 0]
        if (
            positive_gradients.size
            and np.max(positive_gradients) / np.min(positive_gradients) > 100
        ):
            ax_gradient.set_yscale("log")
            ax_gradient.set_ylabel("Gradient Norm (log scale)", fontsize=self.font_size)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)

        if self.save_plot:
            model_stem = os.path.splitext(self.model_name)[0]
            save_path = os.path.join(self.output_dir, f"training_info-{model_stem}.png")
            fig.savefig(save_path, dpi=1000, bbox_inches="tight", facecolor="white")
            if self.verbose:
                print(f"Publication training figure saved to: {save_path}")

        if self.show_plot:
            plt.show()

        return fig


if __name__ == "__main__":
    plotter = EnhancedTrainingPlotter(
        model_file="model.pkl",
        verbose=True,
        font_size=24,
        show_plot=False,
        save_plot=True,
    )
    plotter.plot_training_info_publish(
        loss_y_max=1.5,
        loss_y_min=-0.05,
        train_loss_start_epoch=1,
        val_loss_start_epoch=8,
        gradient_y=3,
        gradient_start_epoch=2,
        show_best_epoch=False,
        plot_raw_training_loss=True,
        convert_to_original_scale=False,
        smooth_val_loss=True,
        smooth_window=5,
    )
