#!/usr/bin/env python3
"""
Script to generate workflow diagram for zeolite ML project - focusing on comparison methodology
Author: Generated for solvation interaction energy prediction study
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np

def create_workflow_diagram():
    """Create a focused diagram showing the three-level comparison approach with left-to-right layout"""
    
    # Create figure with specific size for publication
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'adsorbate': '#E8F4FD',  # Light blue for adsorbate level
        'snapshot': '#FFF2CC',   # Light yellow for snapshot level
        'voxel': '#E1D5E7',      # Light purple for voxel level
        'comparison': '#D5E8D4', # Light green for comparisons
        'border': '#666666',     # Gray for borders
        'dft': '#FFE6E6',        # Light red for DFT
        'ml': '#F0E68C'          # Light gold for ML
    }
    
    # Title
    ax.text(8, 9.5, 'Three-Level Hierarchical Evaluation Strategy', 
            fontsize=16, fontweight='bold', ha='center')
    
    # ===== UPPER PART: Data Flow =====
    ax.text(8, 8.8, 'Data Processing Flow', fontsize=12, fontweight='bold', ha='center', style='italic')
    
    # LEFT: Adsorbate Level (Level 3)
    adsorbate_box = FancyBboxPatch((0.5, 7), 4.5, 1.5,
                                  boxstyle="round,pad=0.15",
                                  facecolor=colors['adsorbate'],
                                  edgecolor=colors['border'], linewidth=2)
    ax.add_patch(adsorbate_box)
    ax.text(2.75, 8.1, 'Adsorbate Level', fontsize=14, fontweight='bold', ha='center')
    ax.text(2.75, 7.8, '1 Adsorbate', fontsize=11, ha='center')
    ax.text(2.75, 7.5, '10 MD snapshots', fontsize=11, ha='center')
    ax.text(2.75, 7.2, '240 total voxel grids', fontsize=11, ha='center', style='italic')
    
    # MIDDLE: Snapshot Level (Level 2)  
    snapshot_box = FancyBboxPatch((5.75, 7), 4.5, 1.5,
                                 boxstyle="round,pad=0.15",
                                 facecolor=colors['snapshot'],
                                 edgecolor=colors['border'], linewidth=2)
    ax.add_patch(snapshot_box)
    ax.text(8, 8.1, 'Snapshot Level', fontsize=14, fontweight='bold', ha='center')
    ax.text(8, 7.8, '1 MD snapshot', fontsize=11, ha='center')
    ax.text(8, 7.5, '24 augmented voxels', fontsize=11, ha='center')
    ax.text(8, 7.2, '1 DFT ΔE_int label', fontsize=11, ha='center', style='italic')
    
    # Add visual representation of snapshots
    for i in range(3):
        x_pos = 6.5 + i * 0.8
        y_pos = 6.3
        snapshot_mini = Rectangle((x_pos, y_pos), 0.6, 0.5,
                                facecolor='white', edgecolor=colors['border'], linewidth=1)
        ax.add_patch(snapshot_mini)
        # Add dots to represent molecules
        for j in range(6):
            dot_x = x_pos + 0.1 + (j % 3) * 0.15
            dot_y = y_pos + 0.1 + (j // 3) * 0.15
            ax.scatter(dot_x, dot_y, c='red' if j % 2 == 0 else 'gray', s=15, alpha=0.7)
    
    ax.text(8, 6.0, '... 10 snapshots total', fontsize=9, ha='center', style='italic')
    
    # RIGHT: Voxel Level (Level 1)
    voxel_box = FancyBboxPatch((11, 7), 4.5, 1.5,
                              boxstyle="round,pad=0.15",
                              facecolor=colors['voxel'],
                              edgecolor=colors['border'], linewidth=2)
    ax.add_patch(voxel_box)
    ax.text(13.25, 8.1, 'Voxel Level (Augmented)', fontsize=14, fontweight='bold', ha='center')
    ax.text(13.25, 7.8, '1 Augmented voxel grid', fontsize=11, ha='center')
    ax.text(13.25, 7.5, '24 rotational variants', fontsize=11, ha='center')
    ax.text(13.25, 7.2, 'per snapshot', fontsize=11, ha='center', style='italic')
    
    # Add visual representation of voxel grid
    for i in range(3):
        for j in range(3):
            x_pos = 12.2 + i * 0.3
            y_pos = 6.3 + j * 0.25
            voxel_mini = Rectangle((x_pos, y_pos), 0.25, 0.2,
                                 facecolor='lightgray', edgecolor='black', linewidth=0.5)
            ax.add_patch(voxel_mini)
    
    # Arrows between levels
    arrow1 = ConnectionPatch((5, 7.75), (5.75, 7.75), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, lw=2)
    ax.add_artist(arrow1)
    ax.text(5.375, 8.0, '×10', fontsize=10, ha='center', fontweight='bold')
    
    arrow2 = ConnectionPatch((10.25, 7.75), (11, 7.75), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, lw=2)
    ax.add_artist(arrow2)
    ax.text(10.625, 8.0, '×24', fontsize=10, ha='center', fontweight='bold')
    
    # ===== LOWER PART: Comparison Methods =====
    ax.text(8, 5.8, 'Performance Evaluation Methods', fontsize=12, fontweight='bold', ha='center', style='italic')
    
    # Level 3 Comparison (Left)
    comp3_box = FancyBboxPatch((0.5, 3.5), 4.5, 2,
                               boxstyle="round,pad=0.15",
                               facecolor=colors['comparison'],
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(comp3_box)
    ax.text(2.75, 5.2, 'Level 3: Adsorbate Evaluation', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.75, 4.9, 'ML Prediction:', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.75, 4.7, 'ΔE_sol^ML (mean of 240)', fontsize=10, ha='center')
    ax.text(2.75, 4.5, 'vs', fontsize=11, ha='center', fontweight='bold', color='red')
    ax.text(2.75, 4.3, 'DFT Target:', fontsize=10, fontweight='bold', ha='center')
    ax.text(2.75, 4.1, 'ΔE_sol^DFT (mean of 10)', fontsize=10, ha='center')
    ax.text(2.75, 3.9, '= Final ΔE_sol', fontsize=10, ha='center', style='italic', color='blue')
    ax.text(2.75, 3.7, 'n = 1 per adsorbate', fontsize=9, ha='center', style='italic')
    
    # Level 2 Comparison (Middle)
    comp2_box = FancyBboxPatch((5.75, 3.5), 4.5, 2,
                               boxstyle="round,pad=0.15",
                               facecolor=colors['comparison'],
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(comp2_box)
    ax.text(8, 5.2, 'Level 2: Snapshot Evaluation', fontsize=12, fontweight='bold', ha='center')
    ax.text(8, 4.9, 'ML Prediction:', fontsize=10, fontweight='bold', ha='center')
    ax.text(8, 4.7, 'ΔE_int^ML (mean of 24)', fontsize=10, ha='center')
    ax.text(8, 4.5, 'vs', fontsize=11, ha='center', fontweight='bold', color='red')
    ax.text(8, 4.3, 'DFT Target:', fontsize=10, fontweight='bold', ha='center')
    ax.text(8, 4.1, 'ΔE_int^DFT', fontsize=10, ha='center')
    ax.text(8, 3.9, '(same snapshot)', fontsize=10, ha='center', style='italic', color='blue')
    ax.text(8, 3.7, 'n = 10 per adsorbate', fontsize=9, ha='center', style='italic')
    
    # Level 1 Comparison (Right)
    comp1_box = FancyBboxPatch((11, 3.5), 4.5, 2,
                               boxstyle="round,pad=0.15",
                               facecolor=colors['comparison'],
                               edgecolor=colors['border'], linewidth=2)
    ax.add_patch(comp1_box)
    ax.text(13.25, 5.2, 'Level 1: Voxel Evaluation', fontsize=12, fontweight='bold', ha='center')
    ax.text(13.25, 4.9, 'ML Prediction:', fontsize=10, fontweight='bold', ha='center')
    ax.text(13.25, 4.7, 'ΔE_int^ML_aug (individual)', fontsize=10, ha='center')
    ax.text(13.25, 4.5, 'vs', fontsize=11, ha='center', fontweight='bold', color='red')
    ax.text(13.25, 4.3, 'DFT Target:', fontsize=10, fontweight='bold', ha='center')
    ax.text(13.25, 4.1, 'ΔE_int^DFT (same snapshot)', fontsize=10, ha='center')
    ax.text(13.25, 3.9, '(same snapshot)', fontsize=10, ha='center', style='italic', color='blue')
    ax.text(13.25, 3.7, 'n = 240 per adsorbate', fontsize=9, ha='center', style='italic')
    
    # DFT and ML Process boxes
    dft_box = FancyBboxPatch((1, 2.5), 6, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['dft'],
                            edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(dft_box)
    ax.text(4, 2.9, 'DFT Ground Truth: 10 ΔE_int values per adsorbate', 
            fontsize=11, fontweight='bold', ha='center')
    
    ml_box = FancyBboxPatch((9, 2.5), 6, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['ml'],
                           edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(ml_box)
    ax.text(12, 2.9, '3D CNN Predictions: 240 values per adsorbate', 
            fontsize=11, fontweight='bold', ha='center')
    
    # ===== ARROWS AND PROCESS FLOW =====
    
    # Vertical arrows from data levels to comparison methods
    arrow3 = ConnectionPatch((2.75, 7), (2.75, 5.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, lw=2, color='blue')
    ax.add_artist(arrow3)
    ax.text(3.2, 6.2, 'Aggregate\n240 preds', fontsize=9, ha='center', fontweight='bold', color='blue')
    
    arrow4 = ConnectionPatch((8, 7), (8, 5.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, lw=2, color='blue')
    ax.add_artist(arrow4)
    ax.text(8.5, 6.2, 'Average\n24 preds', fontsize=9, ha='center', fontweight='bold', color='blue')
    
    arrow5 = ConnectionPatch((13.25, 7), (13.25, 5.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20, lw=2, color='blue')
    ax.add_artist(arrow5)
    ax.text(13.7, 6.2, 'Individual\npred', fontsize=9, ha='center', fontweight='bold', color='blue')
    
    # Arrows from DFT to comparison methods
    arrow6 = ConnectionPatch((4, 2.9), (2.75, 3.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, lw=2, color='red')
    ax.add_artist(arrow6)
    ax.text(3.2, 3.2, 'Mean\n10 ΔE_int', fontsize=9, ha='center', fontweight='bold', color='red')
    
    arrow7 = ConnectionPatch((5.5, 2.9), (8, 3.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, lw=2, color='red')
    ax.add_artist(arrow7)
    ax.text(6.8, 3.2, 'Single\nΔE_int', fontsize=9, ha='center', fontweight='bold', color='red')
    
    arrow8 = ConnectionPatch((9.5, 2.9), (13.25, 3.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, lw=2, color='red')
    ax.add_artist(arrow8)
    ax.text(11.5, 3.2, 'Same\nΔE_int', fontsize=9, ha='center', fontweight='bold', color='red')
    
    # Arrows from ML to comparison methods
    arrow9 = ConnectionPatch((12, 2.9), (2.75, 3.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, lw=2, color='orange')
    ax.add_artist(arrow9)
    ax.text(7, 2.2, 'Mean 240 ML predictions', fontsize=9, ha='center', fontweight='bold', color='orange')
    
    arrow10 = ConnectionPatch((12, 2.9), (8, 3.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, lw=2, color='orange')
    ax.add_artist(arrow10)
    ax.text(10, 3.0, 'Mean 24\nML preds', fontsize=9, ha='center', fontweight='bold', color='orange')
    
    arrow11 = ConnectionPatch((12, 2.9), (13.25, 3.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=15, lw=2, color='orange')
    ax.add_artist(arrow11)
    ax.text(12.8, 3.2, 'Single\nML pred', fontsize=9, ha='center', fontweight='bold', color='orange')
    
    # Add comparison symbols in the boxes
    ax.text(2.75, 4.5, '⚖️', fontsize=20, ha='center')
    ax.text(8, 4.5, '⚖️', fontsize=20, ha='center')
    ax.text(13.25, 4.5, '⚖️', fontsize=20, ha='center')
    
    # Add evaluation metrics labels
    ax.text(0.7, 4.2, 'Metrics:\nMAE, RMSE\nR², etc.', fontsize=8, ha='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    ax.text(5.9, 4.2, 'Metrics:\nMAE, RMSE\nR², etc.', fontsize=8, ha='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    ax.text(11.2, 4.2, 'Metrics:\nMAE, RMSE\nR², etc.', fontsize=8, ha='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    # ===== MATHEMATICAL FORMULATIONS =====
    eq_box = FancyBboxPatch((1, 0.3), 14, 1.5,
                           boxstyle="round,pad=0.1",
                           facecolor='white',
                           edgecolor=colors['border'], linewidth=1)
    ax.add_patch(eq_box)
    
    ax.text(8, 1.6, 'Mathematical Formulations & Evaluation Process', fontsize=12, fontweight='bold', ha='center')
    
    # Add arrows showing the evaluation process
    ax.text(1.5, 1.3, 'Level 3:', fontsize=10, fontweight='bold', color='blue')
    ax.text(3.2, 1.3, 'ΔE_sol^ML ⚖️ ΔE_sol^DFT', fontsize=10)
    ax.text(8.8, 1.3, '→ Final Performance', fontsize=10, fontweight='bold', color='red')
    ax.text(11.5, 1.3, '(Most Important)', fontsize=9, style='italic', color='red')
    
    ax.text(1.5, 1.0, 'Level 2:', fontsize=10, fontweight='bold', color='blue')
    ax.text(3.2, 1.0, 'ΔE_int^ML ⚖️ ΔE_int^DFT', fontsize=10)
    ax.text(7.5, 1.0, 'for each snapshot j', fontsize=9, style='italic')
    ax.text(9.8, 1.0, '(Reduces Noise)', fontsize=9, style='italic', color='green')
    
    ax.text(1.5, 0.7, 'Level 1:', fontsize=10, fontweight='bold', color='blue')
    ax.text(3.2, 0.7, 'ΔE_int^ML_aug ⚖️ ΔE_int^DFT', fontsize=10)
    ax.text(6.5, 0.7, 'for each augmented voxel i', fontsize=9, style='italic')
    ax.text(10.5, 0.7, '(Most Granular)', fontsize=9, style='italic', color='purple')
    
    ax.text(1.5, 0.4, 'Key Insight:', fontsize=10, fontweight='bold', color='red')
    ax.text(3.5, 0.4, 'Each level provides different perspectives on model performance', fontsize=10, color='red')
    
    # Add legend for arrows
    legend_box = FancyBboxPatch((12.5, 0.5), 3, 1.1,
                               boxstyle="round,pad=0.1",
                               facecolor='lightgray',
                               edgecolor=colors['border'], linewidth=1, alpha=0.8)
    ax.add_patch(legend_box)
    ax.text(14, 1.4, 'Arrow Legend:', fontsize=9, fontweight='bold', ha='center')
    ax.text(14, 1.2, '● ML Predictions', fontsize=8, ha='center', color='blue')
    ax.text(14, 1.0, '● DFT Ground Truth', fontsize=8, ha='center', color='red')
    ax.text(14, 0.8, '● ML Processing', fontsize=8, ha='center', color='orange')
    ax.text(14, 0.6, '⚖ Comparison', fontsize=8, ha='center')
    
    plt.tight_layout()
    return fig

def save_diagram(fig, output_dir="../output_figures"):
    """Save the diagram in multiple formats"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in different formats
    formats = ['png', 'pdf', 'svg']
    
    for fmt in formats:
        filename = f"{output_dir}/workflow_diagram.{fmt}"
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved diagram as: {filename}")

if __name__ == "__main__":
    # Create the workflow diagram
    fig = create_workflow_diagram()
    
    # Save the diagram
    save_diagram(fig)
    
    # Display the diagram
    plt.show()
    
    print("Workflow diagram generated successfully!")
    print("\nDiagram explanation:")
    print("1. Data Collection: MD simulations generate snapshots, DFT calculates interaction energies")
    print("2. Data Processing: Convert snapshots to voxel grids, apply 24-fold rotational augmentation")  
    print("3. Machine Learning: 3D CNN predicts interaction energy for each voxel grid")
    print("4. Three-Level Evaluation:")
    print("   - Level 1: Individual prediction vs DFT label")
    print("   - Level 2: Snapshot-averaged prediction vs DFT label")
    print("   - Level 3: Adsorbate-averaged prediction vs DFT solvation energy")
