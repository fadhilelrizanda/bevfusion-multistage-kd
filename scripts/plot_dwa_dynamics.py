import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parents[1]
    history_file = base_dir / 'results' / 'loss_dynamics_history.csv'
    figures_dir = base_dir / 'analysis'
    figures_dir.mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(history_file)
    
    # Define the mapping
    runs = {
        'Student S': '5zkf0n7i',
        'Student XS': '6u81evo4'
    }
    
    weights = {
        'weight_lidar': 'LiDAR Feature',
        'weight_camera': 'Camera Feature',
        'weight_bev': 'BEV Feature',
        'weight_relation': 'Relation',
        'weight_logit': 'Logit'
    }
    
    steps_per_epoch = 308
    
    # Plot style
    plt.rcParams.update({'font.size': 14})
    
    # Use distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for model_name, run_id in runs.items():
        run_df = df[df['run_id'] == run_id].copy()
        
        # Sort by step
        run_df = run_df.sort_values('step')
        
        # Convert steps to epochs
        run_df['epoch'] = run_df['step'] / steps_per_epoch
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for idx, (weight_col, label) in enumerate(weights.items()):
            # Apply rolling mean to smooth out the noisy steps
            # Window size of 50 steps (approx 1/6 of an epoch) for smooth trend
            smoothed = run_df[weight_col].rolling(window=50, min_periods=1).mean()
            
            ax.plot(run_df['epoch'], smoothed, label=label, linewidth=2.5, color=colors[idx], alpha=0.9)
            
        ax.set_xlabel('Epoch')
        ax.set_ylabel('DWA loss weight')
        ax.set_title(f'DWA weight dynamics by epoch ({model_name})', pad=15)
        
        # Add grid lines for readability
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Place legend below the plot to completely avoid right-side clipping
        # and improve aspect ratio when scaled in LaTeX
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=True, shadow=True)
        
        # Ensure the bottom label isn't cut off by explicitly setting bottom margin
        plt.subplots_adjust(bottom=0.35)
        
        filename = f"dwa_weights_{model_name.replace(' ', '_').lower()}.pdf"
        out_path = figures_dir / filename
        # Using pad_inches to provide a safe margin around the whole figure
        plt.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
