#!/usr/bin/env python3
"""
Stylus Wear Animation - Sliding Window Analysis
Creates an animated GIF showing transition analysis across time windows.
Bar widths in summary plot are proportional to interval duration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

# =============================================================================
# CONFIGURATION
# =============================================================================

# Harmonic to analyze: '2H' or '3H'
HARMONIC = '2H'

# Channel to analyze: 'L' or 'R'
CHANNEL = 'L'

# File pattern
FILE_PATTERN = './data/SWS3_VM95ML_REV_T{hours:03d}_{channel}.csv'

# Hours to analyze
HOURS = [0, 48, 96, 192, 240, 288, 384, 480, 576, 672]

# Frequency configuration per harmonic
FREQ_CONFIG = {
    '2H': {
        'min_freq': 1000,
        'max_freq': 10000,
        'column': '2nd Harmonic',
    },
    '3H': {
        'min_freq': 1000,
        'max_freq': 6500,
        'column': '3rd Harmonic',
    },
}

# Sliding window configuration
WINDOW_SIZE = 2      # Number of intervals (time points) in window
WINDOW_STEP = 1      # Step size in intervals

# Animation settings
FRAME_DURATION_MS = 800  # Milliseconds per frame

# Output file (auto-generated if None)
OUTPUT_FILE = None

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(hours_list, channel, file_pattern, freq_min, freq_max):
    """Load all CSV files and filter to frequency range."""
    dfs = {}
    for h in hours_list:
        filepath = file_pattern.format(hours=h, channel=channel)
        try:
            df = pd.read_csv(filepath)
            df = df[(df['Frequency'] >= freq_min) & (df['Frequency'] < freq_max)].copy()
            dfs[h] = df
        except FileNotFoundError:
            print(f"Warning: File not found for T{h:03d}")
    return dfs

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def build_matrix(dfs, hours_list, column):
    """Build frequency x time matrix of harmonic values."""
    freqs = dfs[hours_list[0]]['Frequency'].values
    matrix = []
    for freq in freqs:
        row = []
        for h in hours_list:
            if h in dfs:
                v = dfs[h][dfs[h]['Frequency'] == freq][column].values
                row.append(v[0] if len(v) > 0 else np.nan)
            else:
                row.append(np.nan)
        matrix.append(row)
    return np.array(matrix), freqs

def calculate_window_transitions(matrix, start_idx, window_size):
    """Calculate % increasing transitions for a specific window."""
    pct_per_freq = []
    n_freqs = matrix.shape[0]
    
    for freq_idx in range(n_freqs):
        n_inc = 0
        n_trans = 0
        for i in range(start_idx, start_idx + window_size - 1):
            if i + 1 < matrix.shape[1]:
                v1 = matrix[freq_idx, i]
                v2 = matrix[freq_idx, i + 1]
                if not np.isnan(v1) and not np.isnan(v2):
                    n_trans += 1
                    if v2 > v1:
                        n_inc += 1
        pct = 100 * n_inc / n_trans if n_trans > 0 else 50
        pct_per_freq.append(pct)
    
    return pct_per_freq

def precompute_all_windows(matrix, hours_list, window_size, window_step):
    """Precompute transition stats for all windows."""
    n_windows = (len(hours_list) - window_size) // window_step + 1
    window_stats = []
    
    for w in range(n_windows):
        start_idx = w * window_step
        end_idx = start_idx + window_size - 1
        pct_vals = calculate_window_transitions(matrix, start_idx, window_size)
        mean_val = np.mean(pct_vals)
        
        # Calculate interval duration in hours
        duration = hours_list[end_idx] - hours_list[start_idx]
        
        window_stats.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_hours': hours_list[start_idx],
            'end_hours': hours_list[end_idx],
            'duration': duration,
            'pct_vals': pct_vals,
            'mean': mean_val
        })
    
    return window_stats

# =============================================================================
# ANIMATION (Frame-by-frame rendering)
# =============================================================================

def create_animation(hours_list, matrix, freqs, window_size, window_step, output_file, frame_duration, channel, harmonic, config):
    """Create animated GIF with sliding window using frame-by-frame rendering."""
    
    # Precompute all window stats
    window_stats = precompute_all_windows(matrix, hours_list, window_size, window_step)
    n_windows = len(window_stats)
    all_means = [ws['mean'] for ws in window_stats]
    all_durations = [ws['duration'] for ws in window_stats]
    
    # Calculate bar positions and widths proportional to duration
    # Normalize so total width spans the plot
    total_duration = sum(all_durations)
    bar_widths = [d / total_duration * n_windows * 0.9 for d in all_durations]
    
    # Calculate cumulative positions (left edge of each bar)
    bar_positions = []
    cumsum = 0
    for d in all_durations:
        bar_positions.append(cumsum / total_duration * n_windows)
        cumsum += d
    
    # Center positions for line plot
    bar_centers = [bar_positions[i] + bar_widths[i] / 2 for i in range(n_windows)]
    
    tick_spacing = max(1, len(freqs) // 10)
    
    frames = []
    
    for frame in range(n_windows):
        ws = window_stats[frame]
        
        # Create fresh figure for each frame
        fig = plt.figure(figsize=(14, 10))
        
        ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=2)  # Heatmap
        ax2 = plt.subplot2grid((4, 2), (0, 1), rowspan=2)  # Bar chart
        ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=2, rowspan=2)  # Summary
        
        fig.suptitle(f'{channel} CHANNEL - {harmonic} Sliding Window Transition Analysis', fontsize=14, fontweight='bold')
        
        # Panel 1: Heatmap (drawn fresh each frame, identical each time)
        im = ax1.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=-50, vmax=-20, interpolation='nearest')
        ax1.set_xticks(range(len(hours_list)))
        ax1.set_xticklabels(hours_list)
        ax1.set_yticks(range(0, len(freqs), tick_spacing))
        ax1.set_yticklabels([f'{int(freqs[i])}' for i in range(0, len(freqs), tick_spacing)])
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title(f'{harmonic} Distortion Heatmap ({config["min_freq"]/1000:.0f}-{config["max_freq"]/1000:.1f} kHz)')
        plt.colorbar(im, ax=ax1, label=f'{harmonic} (dB)')
        
        # Window highlight rectangle
        rect = patches.Rectangle(
            (ws['start_idx'] - 0.5, -0.5), 
            window_size, 
            len(freqs),
            linewidth=3, edgecolor='white', facecolor='none', linestyle='-'
        )
        ax1.add_patch(rect)
        
        # Panel 2: Transition bar chart
        ax2.bar(range(len(freqs)), ws['pct_vals'], width=1.0, color='steelblue', alpha=0.7)
        ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='Random (50%)')
        ax2.axhline(y=ws['mean'], color='green', linestyle='-', linewidth=2, label='Mean')
        
        ax2.set_xticks(range(0, len(freqs), tick_spacing))
        ax2.set_xticklabels([f'{int(freqs[i])}' for i in range(0, len(freqs), tick_spacing)], rotation=45)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('% Transitions Increasing')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_title(f"Window: T{ws['start_hours']:03d}-T{ws['end_hours']:03d} ({ws['duration']}h) (Mean: {ws['mean']:.1f}%)")
        
        # Panel 3: Summary with proportional bar widths
        ax3.set_xlim(-0.2, n_windows + 0.2)
        ax3.set_ylim(0, 100)
        
        # Create tick labels with duration
        window_labels = [f"T{ws2['start_hours']:03d}-T{ws2['end_hours']:03d}\n({ws2['duration']}h)" for ws2 in window_stats]
        ax3.set_xticks(bar_centers)
        ax3.set_xticklabels(window_labels, fontsize=8)
        ax3.set_xlabel('Window (bar width ∝ interval duration)')
        ax3.set_ylabel('Mean % Increasing')
        ax3.set_title('Summary: Mean Transition % by Window')
        ax3.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Random (50%)')
        ax3.grid(True, alpha=0.3)
        
        # Draw bars up to current frame with proportional widths
        for i in range(frame + 1):
            color = 'forestgreen' if all_means[i] >= 60 else ('indianred' if all_means[i] < 50 else 'gold')
            ax3.bar(bar_centers[i], all_means[i], width=bar_widths[i], color=color, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
        
        # Draw line connecting points
        if frame > 0:
            ax3.plot(bar_centers[:frame + 1], all_means[:frame + 1], 'o-', color='darkblue', linewidth=2, markersize=8)
        
        # Current marker
        ax3.plot(bar_centers[frame], ws['mean'], 'o', color='lime', markersize=15, markeredgecolor='black', markeredgewidth=2)
        
        ax3.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # Render to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf).copy()
        frames.append(img)
        buf.close()
        plt.close(fig)
    
    # Save as GIF
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0
    )
    
    print(f"Saved animation to {output_file}")
    
    # Print summary table
    print(f"\nWindow Summary:")
    print(f"{'Window':<15} {'Duration':<10} {'Mean %':<10} {'Signal'}")
    print("-" * 45)
    for ws in window_stats:
        signal = "CLEAR" if ws['mean'] >= 60 else ("noise" if ws['mean'] >= 45 else "inverse?")
        print(f"T{ws['start_hours']:03d}-T{ws['end_hours']:03d}     {ws['duration']}h        {ws['mean']:5.1f}%     {signal}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    config = FREQ_CONFIG[HARMONIC]
    column = config['column']
    
    # Auto-generate output filename if not specified
    output_file = OUTPUT_FILE
    if output_file is None:
        output_file = f'./outputs/{HARMONIC}_wear_animation_{CHANNEL}.gif'
    
    dfs = load_data(HOURS, CHANNEL, FILE_PATTERN, config['min_freq'], config['max_freq'])
    
    if not dfs:
        print("No data loaded!")
        return
    
    matrix, freqs = build_matrix(dfs, HOURS, column)
    
    print(f"Harmonic: {HARMONIC} ({column})")
    print(f"Data: {len(freqs)} frequencies, {len(HOURS)} time points")
    print(f"Frequency range: {config['min_freq']}-{config['max_freq']} Hz")
    print(f"Window: {WINDOW_SIZE} intervals, step {WINDOW_STEP}")
    print(f"Generating animation...")
    
    create_animation(HOURS, matrix, freqs, WINDOW_SIZE, WINDOW_STEP, output_file, FRAME_DURATION_MS, CHANNEL, HARMONIC, config)

if __name__ == '__main__':
    main()
