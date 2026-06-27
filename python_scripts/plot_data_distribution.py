"""
Visualize the DFT interaction-energy dataset used in this study.

This script reads the energy-label CSV files stored in ``database/``, combines
them into one pandas DataFrame, derives solvent and pore-type labels from the
``environment`` column, and generates figures that describe the energy
distribution and MD snapshot variation. Each CSV row represents one MD
snapshot and contains its DFT interaction energy (``intE``) and the corresponding
system-averaged energy (``intE_avg``).

The :class:`DataVisualizer` class provides the following methods:

``__init__``
    Define the input and output paths, plotting options, and figure resolution,
    then call ``read_data`` to load the dataset.

``read_data``
    Read all CSV files matching ``csv_pattern``, combine them into one DataFrame,
    derive ``solvent`` and ``pore_type`` columns, apply the existing data filter,
    and print a summary of the loaded dataset.

``plot_intE_distribution_for_publication``
    Generate the three-panel Figure 2 used in the manuscript: the snapshot-level
    DFT interaction-energy distribution, the system-averaged DFT energy
    distribution, and the relationship between these two energy quantities.
    The figure is saved as ``figure_2_dft_energy_distribution.png``.

``plot_boxplot_by_pore``
    Compare snapshot-level interaction energies by pore type and by methanol
    concentration. The figure is saved as
    ``dft_interaction_energy_boxplots.png``.

``plot_violin_by_pore``
    Show the density and quartiles of snapshot-level interaction energies for
    hydrophilic and hydrophobic pores. The figure is saved as
    ``dft_interaction_energy_violin_by_pore.png``.

``plot_kde_by_pore``
    Compare kernel density estimates of snapshot-level interaction energies for
    the two pore types. The figure is saved as
    ``dft_interaction_energy_kde_by_pore.png``.

``plot_qq_plot_intE``
    Generate a normal Q-Q plot for the snapshot-level DFT interaction energies.
    The figure is saved as ``dft_interaction_energy_qq_plot.png``.

``plot_scatter_intE_vs_intE_avg``
    Plot each snapshot-level interaction energy against its corresponding
    system-averaged energy, with points colored by pore type. The figure is saved
    as ``snapshot_vs_system_average_energy.png``.

``plot_md_sampling_analysis``
    Generate a four-panel summary of MD snapshot trajectories, per-system energy
    standard deviations, variation versus mean energy, and energy ranges. The
    figure is saved as ``md_snapshot_sampling_analysis.png``.

When ``save_fig`` is enabled, all figures are written to
``output_figures/data_distribution/``. When ``show_plot`` is enabled, each
figure is also displayed interactively.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from core.path import get_paths


class DataVisualizer:
    """
    Load the DFT energy dataset and generate distribution and sampling figures.

    The combined dataset is stored in ``self.data``. Plotting behavior is
    controlled by ``save_fig`` and ``show_plot``, while all output locations are
    obtained from :func:`core.path.get_paths`.
    """

    def __init__(
        self,
        csv_pattern: str = "*.csv",
        save_fig: bool = False,
        output_dir_key: str = "output_figure_path",
        font_size: int = 14,
        show_plot: bool = True,
    ):
        """
        Initialize the data visualizer and load the energy-label dataset.

        Parameters
        ----------
        csv_pattern : str, default="*.csv"
            Filename pattern used to select CSV files from the database
            directory.
        save_fig : bool, default=False
            Save generated figures when True.
        output_dir_key : str, default="output_figure_path"
            Key passed to ``get_paths`` to locate the base figure directory.
        font_size : int, default=14
            Base font size used by the plotting methods.
        show_plot : bool, default=True
            Display generated figures interactively when True.
        """

        # Paths and plotting configuration
        self.data_dir = get_paths("database_path")
        base_output_dir = get_paths(output_dir_key)
        self.output_dir = os.path.join(base_output_dir, "data_distribution")
        self.csv_pattern = csv_pattern
        self.save_fig = save_fig
        self.dpi = 1000  # High resolution for figures
        self.font_size = font_size
        self.show_plot = show_plot

        # Dataset loaded and prepared by read_data
        self.data = None
        self.read_data()

    def read_data(self):
        """
        Read, combine, and prepare the interaction-energy CSV files.

        All files matching ``self.csv_pattern`` in ``self.data_dir`` are loaded
        and concatenated. The method records the source filename, parses the
        solvent and pore type from ``environment``, applies the existing filter
        for the known erroneous adsorbate entry, and prints dataset counts.

        The prepared DataFrame is assigned to ``self.data``. This method does not
        return a separate value.
        """
        import glob

        # Get all CSV files matching the pattern
        csv_files = glob.glob(os.path.join(self.data_dir, self.csv_pattern))

        if not csv_files:
            raise ValueError(
                f"No CSV files found in {self.data_dir} matching pattern {self.csv_pattern}"
            )

        print(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")

        # Read and combine all CSV files
        dataframes = []
        for file in csv_files:
            df = pd.read_csv(file)
            df["source_file"] = os.path.basename(file)  # Add source file info
            dataframes.append(df)
            print(f"  Loaded {len(df)} rows from {os.path.basename(file)}")

        # Combine all dataframes
        self.data = pd.concat(dataframes, ignore_index=True)

        # Extract pore_type from environment column
        # environment format: "solvent-pore_type" (e.g., "methanol_240_water_960-hydrophobic")
        self.data["pore_type"] = self.data["environment"].str.split("-").str[1]

        # Extract solvent info as well (optional)
        self.data["solvent"] = self.data["environment"].str.split("-").str[0]

        print(f"\nTotal combined dataset before filtering: {len(self.data)} rows")

        # Remove specific problematic adsorbate: water_pure-hydrophilic with 05_3c_aldehyde_wrong
        rows_before = len(self.data)
        self.data = self.data[
            ~(
                (self.data["environment"] == "water_pure-hydrophilic")
                & (self.data["adsorbate"] == "05_3c_aldehyde_wrong")
            )
        ]
        rows_removed = rows_before - len(self.data)

        if rows_removed > 0:
            print(
                f"Removed {rows_removed} rows for adsorbate '05_3c_aldehyde_wrong' in 'water_pure-hydrophilic' environment"
            )

        print(f"\nTotal combined dataset after filtering: {len(self.data)} rows")
        print(f"Unique zeolites: {self.data['zeolite'].unique()}")
        print(f"Unique pore types: {self.data['pore_type'].unique()}")
        print(f"Unique adsorbates: {self.data['adsorbate'].nunique()}")
        print(f"Unique environments: {self.data['environment'].nunique()}")
        print(
            f"Unique env-adsorbate combinations (ΔE_sol values): {self.data.groupby(['environment', 'adsorbate']).ngroups}"
        )

    def plot_intE_distribution_for_publication(self, bins: int = 30, figsize=(21, 7)):
        """
        Generate the manuscript figure for the DFT energy distributions.

        Panel (a) shows the distribution of snapshot-level DFT interaction
        energies. Panel (b) shows the distribution of the associated
        system-averaged energies. Panel (c) compares the two quantities and adds
        a diagonal reference line and a smoothed, system-dependent standard-
        deviation region.

        Parameters
        ----------
        bins : int, default=30
            Number of histogram bins used in panels (a) and (b).
        figsize : tuple, default=(21, 7)
            Matplotlib figure size in inches.

        Output
        ------
        figure_2_dft_energy_distribution.png
            Saved in ``self.output_dir`` when ``self.save_fig`` is True.
        """
        sns.set_style("white")
        intE = self.data["intE"]
        intE_avg = self.data["intE_avg"]

        # Compute statistics for intE
        stats_intE = {
            "min": intE.min(),
            "max": intE.max(),
            "mean": intE.mean(),
            "median": intE.median(),
            "std": intE.std(),
            "q1": intE.quantile(0.25),
            "q3": intE.quantile(0.75),
        }
        print("intE statistics:")
        for k, v in stats_intE.items():
            print(f"  {k}: {v:.3f}")

        # Compute statistics for intE_avg (delta E_sol)
        stats_intE_avg = {
            "min": intE_avg.min(),
            "max": intE_avg.max(),
            "mean": intE_avg.mean(),
            "median": intE_avg.median(),
            "std": intE_avg.std(),
            "q1": intE_avg.quantile(0.25),
            "q3": intE_avg.quantile(0.75),
        }
        print("\nintE_avg (delta E_sol) statistics:")
        for k, v in stats_intE_avg.items():
            print(f"  {k}: {v:.3f}")

        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=self.dpi)

        # Panel (a): snapshot-level interaction-energy distribution
        sns.histplot(
            intE,
            bins=bins,
            kde=True,
            color="#1f78b4",
            edgecolor="black",
            alpha=0.7,
            ax=axes[0],
            kde_kws={"bw_adjust": 2},  # Smoother KDE
        )
        axes[0].tick_params(axis="both", which="major", labelsize=self.font_size)
        axes[0].set_xlabel(
            r"$\Delta E^{\mathit{DFT}}_{\mathit{int}}$ (eV)", fontsize=self.font_size
        )
        axes[0].set_ylabel("Count", fontsize=self.font_size)
        axes[0].text(
            -0.18,
            1.0,
            "(a)",
            transform=axes[0].transAxes,
            fontsize=self.font_size,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        # Panel (b): system-averaged interaction-energy distribution
        sns.histplot(
            intE_avg,
            bins=bins,
            kde=True,
            color="#ff7f0e",
            edgecolor="black",
            alpha=0.7,
            ax=axes[1],
            kde_kws={"bw_adjust": 2},  # Smoother KDE
        )
        axes[1].tick_params(axis="both", which="major", labelsize=self.font_size)
        axes[1].set_xlabel(
            r"$\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)", fontsize=self.font_size
        )
        axes[1].set_ylabel("Count", fontsize=self.font_size)
        axes[1].text(
            -0.23,
            1.0,
            "(b)",
            transform=axes[1].transAxes,
            fontsize=self.font_size,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        # Panel (c): snapshot-level versus system-averaged interaction energy
        sns.scatterplot(
            data=self.data,
            x="intE_avg",
            y="intE",
            alpha=0.6,
            ax=axes[2],
            color="#2ca02c",
            legend=False,
            s=50,
        )
        axes[2].set_xlabel(
            r"$\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)", fontsize=self.font_size
        )
        axes[2].set_ylabel(
            r"$\Delta E^{\mathit{DFT}}_{\mathrm{int}}$ (eV)", fontsize=self.font_size
        )
        axes[2].tick_params(axis="both", which="major", labelsize=self.font_size)
        axes[2].text(
            -0.23,
            1.0,
            "(c)",
            transform=axes[2].transAxes,
            fontsize=self.font_size,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        # Calculate a dynamic standard-deviation region from each system
        # Each system has 10 snapshots with the same intE_avg value
        grouped = self.data.groupby("intE_avg")["intE"].agg(["mean", "std"])
        grouped = grouped.sort_index()  # Sort by intE_avg values

        # Extract the system-average energy, snapshot mean, and standard deviation
        x_unique = grouped.index.values
        y_mean = grouped["mean"].values
        y_std = grouped["std"].values

        # Extend the range by 0.1 eV on both ends for smoother visualization
        x_min_extended = x_unique.min() - 0.1
        x_max_extended = x_unique.max() + 0.1

        # Create an extended x range for interpolation
        x_extended = np.linspace(x_min_extended, x_max_extended, 200)

        # Interpolate the mean and standard deviation over the extended range
        from scipy.interpolate import interp1d
        from scipy.signal import savgol_filter

        # Use linear extrapolation at the boundaries
        interp_mean = interp1d(x_unique, y_mean, kind="linear", fill_value="extrapolate")
        interp_std = interp1d(x_unique, y_std, kind="linear", fill_value="extrapolate")

        y_mean_extended = interp_mean(x_extended)
        y_std_extended = interp_std(x_extended)

        # Smooth the standard-deviation curve using a Savitzky-Golay filter
        # The window length must be odd and shorter than the data array
        window_length = min(51, len(y_std_extended) // 2 * 2 - 1)  # Ensure odd number
        if window_length >= 5:  # Only smooth if we have enough points
            y_std_extended_smooth = savgol_filter(
                y_std_extended, window_length=window_length, polyorder=3
            )
        else:
            y_std_extended_smooth = y_std_extended

        # Add a y = x reference line
        lims = [
            np.min([axes[2].get_xlim(), axes[2].get_ylim()]),
            np.max([axes[2].get_xlim(), axes[2].get_ylim()]),
        ]
        x_range = np.linspace(lims[0], lims[1], 100)
        axes[2].plot(x_range, x_range, "k--", alpha=0.3, zorder=0, linewidth=2)

        # Plot the smoothed, dynamic +/-1 standard-deviation region
        axes[2].fill_between(
            x_extended,
            y_mean_extended - y_std_extended_smooth,
            y_mean_extended + y_std_extended_smooth,
            color="#2ca02c",
            alpha=0.2,
            zorder=0,
        )

        # Print the average within-system standard deviation for reference
        mean_std = y_std.mean()
        print(f"\nAverage per-adsorbate standard deviation: {mean_std:.4f} eV")

        fig.tight_layout()
        if self.save_fig:
            os.makedirs(self.output_dir, exist_ok=True)
            fig_path = os.path.join(self.output_dir, "figure_2_dft_energy_distribution.png")
            fig.savefig(fig_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Figure saved to {fig_path}")
        if self.show_plot:
            plt.show()
        plt.close(fig)

    def plot_boxplot_by_pore(self, figsize=(18, 7)):
        """
        Compare DFT interaction energies by pore and solvent environment.

        The left panel groups all snapshot-level energies by hydrophilic or
        hydrophobic pore type. The right panel separates the same energies by
        methanol concentration and uses color to distinguish pore type.

        Parameters
        ----------
        figsize : tuple, default=(18, 7)
            Matplotlib figure size in inches.

        Output
        ------
        dft_interaction_energy_boxplots.png
            Saved in ``self.output_dir`` when ``self.save_fig`` is True.
        """
        sns.set_style("white")
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=self.dpi)
        auxiliary_font_size = min(self.font_size, 20)
        tick_font_size = min(self.font_size, 18)

        # Use consistent colors for the two pore types
        colors = ["#1f77b4", "#ff7f0e"]  # Blue for hydrophilic, Orange for hydrophobic

        # Left panel: interaction energy grouped by pore type
        sns.boxplot(
            x="pore_type",
            y="intE",
            hue="pore_type",
            data=self.data,
            ax=axes[0],
            palette=colors,
            legend=False,
        )
        axes[0].set_xlabel("Pore Type", fontsize=auxiliary_font_size)
        axes[0].set_ylabel("Interaction Energy (eV)", fontsize=auxiliary_font_size)
        axes[0].set_title("Interaction Energy by Pore Type", fontsize=auxiliary_font_size)
        axes[0].tick_params(axis="both", which="major", labelsize=tick_font_size)

        # Map simulation directory names to readable methanol concentrations
        solvent_mapping = {
            "water_pure": "0% MeOH",
            "methanol_120_water_1080": "10% MeOH",
            "methanol_240_water_960": "20% MeOH",
            "methanol_600_water_600": "50% MeOH",
        }

        # Preserve the original DataFrame while adding display labels
        data_renamed = self.data.copy()
        data_renamed["solvent_percent"] = data_renamed["solvent"].map(solvent_mapping)

        # Display solvent compositions in increasing methanol concentration
        solvent_order = ["0% MeOH", "10% MeOH", "20% MeOH", "50% MeOH"]

        # Right panel: interaction energy grouped by solvent and pore type
        sns.boxplot(
            x="solvent_percent",
            y="intE",
            hue="pore_type",
            data=data_renamed,
            ax=axes[1],
            palette=colors,
            order=solvent_order,
        )
        axes[1].set_xlabel("Methanol Concentration", fontsize=auxiliary_font_size)
        axes[1].set_ylabel("Interaction Energy (eV)", fontsize=auxiliary_font_size)
        axes[1].set_title("Interaction Energy by Solvent Composition", fontsize=auxiliary_font_size)
        axes[1].tick_params(axis="both", which="major", labelsize=tick_font_size)
        axes[1].tick_params(axis="x", rotation=0, labelsize=tick_font_size)
        legend = axes[1].get_legend()
        legend.set_title("Pore Type", prop={"size": 20})
        for text in legend.get_texts():
            text.set_fontsize(18)

        fig.tight_layout()
        if self.save_fig:
            os.makedirs(self.output_dir, exist_ok=True)
            fig_path = os.path.join(self.output_dir, "dft_interaction_energy_boxplots.png")
            fig.savefig(fig_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Figure saved to {fig_path}")
        if self.show_plot:
            plt.show()
        plt.close(fig)

    def plot_violin_by_pore(self, figsize=(8, 6)):
        """
        Plot interaction-energy density and quartiles by pore type.

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            Matplotlib figure size in inches.

        Output
        ------
        dft_interaction_energy_violin_by_pore.png
            Saved in ``self.output_dir`` when ``self.save_fig`` is True.
        """
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        auxiliary_font_size = min(self.font_size, 20)
        sns.violinplot(x="pore_type", y="intE", data=self.data, inner="quartile", ax=ax)
        ax.set_xlabel("Pore Type", fontsize=auxiliary_font_size)
        ax.set_ylabel(
            r"$\Delta E^{\mathit{DFT}}_{\mathit{int}}$ (eV)", fontsize=auxiliary_font_size
        )
        ax.set_title("DFT Interaction Energy by Pore Type", fontsize=auxiliary_font_size)
        ax.tick_params(axis="both", which="major", labelsize=auxiliary_font_size - 2)
        fig.tight_layout()
        if self.save_fig:
            os.makedirs(self.output_dir, exist_ok=True)
            fig_path = os.path.join(self.output_dir, "dft_interaction_energy_violin_by_pore.png")
            fig.savefig(fig_path, dpi=self.dpi)
            print(f"Figure saved to {fig_path}")
        if self.show_plot:
            plt.show()
        plt.close(fig)

    def plot_kde_by_pore(self, figsize=(8, 6)):
        """
        Compare interaction-energy kernel density estimates by pore type.

        Each density is normalized independently because ``common_norm`` is
        disabled, allowing the shapes of the hydrophilic and hydrophobic energy
        distributions to be compared directly.

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            Matplotlib figure size in inches.

        Output
        ------
        dft_interaction_energy_kde_by_pore.png
            Saved in ``self.output_dir`` when ``self.save_fig`` is True.
        """
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        auxiliary_font_size = min(self.font_size, 20)
        sns.kdeplot(
            data=self.data,
            x="intE",
            hue="pore_type",
            fill=True,
            common_norm=False,
            alpha=0.5,
            ax=ax,
        )
        ax.set_xlabel(
            r"$\Delta E^{\mathit{DFT}}_{\mathit{int}}$ (eV)", fontsize=auxiliary_font_size
        )
        ax.set_ylabel("Density", fontsize=auxiliary_font_size)
        ax.set_title("DFT Interaction Energy Density by Pore Type", fontsize=auxiliary_font_size)
        ax.tick_params(axis="both", which="major", labelsize=auxiliary_font_size - 2)
        legend = ax.get_legend()
        legend.set_title("Pore Type", prop={"size": 20})
        for text in legend.get_texts():
            text.set_fontsize(18)
        fig.tight_layout()
        if self.save_fig:
            os.makedirs(self.output_dir, exist_ok=True)
            fig_path = os.path.join(self.output_dir, "dft_interaction_energy_kde_by_pore.png")
            fig.savefig(fig_path, dpi=self.dpi)
            print(f"Figure saved to {fig_path}")
        if self.show_plot:
            plt.show()
        plt.close(fig)

    def plot_qq_plot_intE(self, figsize=(7, 6)):
        """
        Generate a normal Q-Q plot of snapshot-level interaction energies.

        The ordered ``intE`` values are compared with theoretical quantiles from
        a normal distribution using ``scipy.stats.probplot``.

        Parameters
        ----------
        figsize : tuple, default=(7, 6)
            Matplotlib figure size in inches.

        Output
        ------
        dft_interaction_energy_qq_plot.png
            Saved in ``self.output_dir`` when ``self.save_fig`` is True.
        """
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        auxiliary_font_size = min(self.font_size, 20)
        stats.probplot(self.data["intE"], dist="norm", plot=ax)
        ax.set_title("Q-Q Plot of DFT Interaction Energy", fontsize=auxiliary_font_size)
        ax.set_xlabel("Theoretical Quantiles", fontsize=auxiliary_font_size)
        ax.set_ylabel("Ordered Interaction Energies (eV)", fontsize=auxiliary_font_size)
        ax.tick_params(axis="both", which="major", labelsize=auxiliary_font_size - 2)
        fig.tight_layout()
        if self.save_fig:
            os.makedirs(self.output_dir, exist_ok=True)
            fig_path = os.path.join(self.output_dir, "dft_interaction_energy_qq_plot.png")
            fig.savefig(fig_path, dpi=self.dpi)
            print(f"Figure saved to {fig_path}")
        if self.show_plot:
            plt.show()
        plt.close(fig)

    def plot_scatter_intE_vs_intE_avg(self, figsize=(8, 6)):
        """
        Compare snapshot-level and system-averaged interaction energies.

        Each point represents one MD snapshot. The x-coordinate is the
        system-averaged DFT energy, the y-coordinate is the snapshot-level DFT
        energy, and color indicates the pore type. A y = x line is included as a
        visual reference.

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            Matplotlib figure size in inches.

        Output
        ------
        snapshot_vs_system_average_energy.png
            Saved in ``self.output_dir`` when ``self.save_fig`` is True.
        """
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        auxiliary_font_size = min(self.font_size, 20)
        sns.scatterplot(data=self.data, x="intE_avg", y="intE", hue="pore_type", alpha=0.6, ax=ax)
        ax.set_xlabel(
            r"$\Delta E^{\mathit{DFT}}_{\mathit{sol}}$ (eV)", fontsize=auxiliary_font_size
        )
        ax.set_ylabel(
            r"$\Delta E^{\mathit{DFT}}_{\mathit{int}}$ (eV)", fontsize=auxiliary_font_size
        )
        ax.set_title("Snapshot vs System-Averaged Interaction Energy", fontsize=auxiliary_font_size)
        ax.tick_params(axis="both", which="major", labelsize=auxiliary_font_size - 2)
        legend = ax.get_legend()
        legend.set_title("Pore Type", prop={"size": 20})
        for text in legend.get_texts():
            text.set_fontsize(18)

        # Add a y = x reference line
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, "k--", alpha=0.3, zorder=0)

        fig.tight_layout()
        if self.save_fig:
            os.makedirs(self.output_dir, exist_ok=True)
            fig_path = os.path.join(self.output_dir, "snapshot_vs_system_average_energy.png")
            fig.savefig(fig_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Figure saved to {fig_path}")
        if self.show_plot:
            plt.show()
        plt.close(fig)

    def plot_md_sampling_analysis(self, figsize=(18, 12)):
        """
        Summarize energy variation across the sampled MD snapshots.

        The four panels show: (1) energy trajectories across snapshots for the
        selected adsorbates and environments, (2) the distribution of
        within-system standard deviations, (3) standard deviation versus mean
        interaction energy, and (4) within-system energy range by pore type.

        Parameters
        ----------
        figsize : tuple, default=(18, 12)
            Matplotlib figure size in inches.

        Output
        ------
        md_snapshot_sampling_analysis.png
            Saved in ``self.output_dir`` when ``self.save_fig`` is True.
        """
        sns.set_style("white")
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=self.dpi)
        auxiliary_font_size = min(self.font_size, 18)

        # Panel 1: energy trajectories for selected adsorbates and environments
        selected_adsorbates = self.data["adsorbate"].value_counts().head(8).index
        for i, ads in enumerate(selected_adsorbates):
            subset = self.data[self.data["adsorbate"] == ads]
            if len(subset) > 0:
                for env in subset["environment"].unique():
                    env_data = subset[subset["environment"] == env].sort_values("snapshot")
                    axes[0, 0].plot(
                        env_data["snapshot"],
                        env_data["intE"],
                        marker="o",
                        alpha=0.7,
                        linewidth=1,
                        markersize=3,
                    )

        axes[0, 0].set_title("MD Snapshot Energy Trajectories", fontsize=auxiliary_font_size)
        axes[0, 0].set_xlabel("Snapshot Number", fontsize=auxiliary_font_size)
        axes[0, 0].set_ylabel("Interaction Energy (eV)", fontsize=auxiliary_font_size)

        # Panel 2: distribution of within-system energy standard deviations
        variance_analysis = (
            self.data.groupby(["environment", "adsorbate"])
            .agg({"intE": ["std", "mean", "min", "max"]})
            .round(3)
        )
        variance_analysis.columns = ["std", "mean", "min", "max"]
        variance_analysis["range"] = variance_analysis["max"] - variance_analysis["min"]
        variance_analysis = variance_analysis.reset_index()

        sns.histplot(variance_analysis["std"], bins=15, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title("Energy Standard Deviations", fontsize=auxiliary_font_size)
        axes[0, 1].set_xlabel("Standard Deviation (eV)", fontsize=auxiliary_font_size)
        axes[0, 1].set_ylabel("Count", fontsize=auxiliary_font_size)

        # Panel 3: relationship between mean energy and sampling variation
        pore_colors = variance_analysis["environment"].str.split("-").str[1]
        sns.scatterplot(data=variance_analysis, x="mean", y="std", hue=pore_colors, ax=axes[1, 0])
        axes[1, 0].set_title("Sampling Variation vs Mean Energy", fontsize=auxiliary_font_size)
        axes[1, 0].set_xlabel("Mean Interaction Energy (eV)", fontsize=auxiliary_font_size)
        axes[1, 0].set_ylabel("Standard Deviation (eV)", fontsize=auxiliary_font_size)

        # Panel 4: within-system energy ranges grouped by pore type
        sns.boxplot(
            data=variance_analysis,
            y="range",
            x=variance_analysis["environment"].str.split("-").str[1],
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Energy Range Across Snapshots", fontsize=auxiliary_font_size)
        axes[1, 1].set_xlabel("Pore Type", fontsize=auxiliary_font_size)
        axes[1, 1].set_ylabel("Energy Range (eV)", fontsize=auxiliary_font_size)

        for ax in axes.flat:
            ax.tick_params(axis="both", which="major", labelsize=auxiliary_font_size - 2)

        plt.tight_layout()
        if self.save_fig:
            os.makedirs(self.output_dir, exist_ok=True)
            fig_path = os.path.join(self.output_dir, "md_snapshot_sampling_analysis.png")
            fig.savefig(fig_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Figure saved to {fig_path}")
        if self.show_plot:
            plt.show()
        plt.close(fig)


# Example usage
if __name__ == "__main__":
    # Read all CSV files in database directory
    viz = DataVisualizer(
        csv_pattern="*.csv",
        font_size=26,
        show_plot=True,
        save_fig=True,
    )

    # Plot distribution of interaction energy
    viz.plot_intE_distribution_for_publication()

    # Plot boxplot of interaction energy by pore type
    viz.plot_boxplot_by_pore()

    # Plot violin plot of interaction energy by pore type
    viz.plot_violin_by_pore()

    # Plot KDE by pore type
    viz.plot_kde_by_pore()

    # Plot Q-Q plot of intE
    viz.plot_qq_plot_intE()

    # Plot scatter plot of intE vs intE_avg
    viz.plot_scatter_intE_vs_intE_avg()

    # MD sampling characteristics
    viz.plot_md_sampling_analysis()

    # Print summary statistics for the paper
    print("\n" + "=" * 50)
    print("DATASET SUMMARY FOR PAPER")
    print("=" * 50)
    print(f"Total snapshots: {len(viz.data)}")
    print(f"Unique adsorbates: {viz.data['adsorbate'].nunique()}")
    print(f"Unique environments: {viz.data['environment'].nunique()}")
    print(f"Snapshots per adsorbate: {len(viz.data) // viz.data['adsorbate'].nunique()}")
    print(f"Energy range: {viz.data['intE'].min():.3f} to {viz.data['intE'].max():.3f} eV")
    print(f"Mean energy: {viz.data['intE'].mean():.3f} ± {viz.data['intE'].std():.3f} eV")
    print(f"Pore types: {', '.join(viz.data['pore_type'].unique())}")
