#!/usr/bin/env python3
"""
Distortion Animation - Sliding Window Analysis
Creates an animated GIF showing transition analysis across time windows.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# =============================================================================
# CONFIGURATION
# =============================================================================

# Channel to analyze: 'L' or 'R'
CHANNEL = 'L'

# File pattern
FILE_PATTERN = 'SWS3_VM95ML_REV_T{hours:03d}_{channel}.csv'

# Hours to analyze
HOURS = [0, 48, 96, 192, 240, 288, 384, 480, 576, 672]

# Frequency range for analysis (Hz)
FREQ_MIN = 1000
FREQ_MAX = 10000

# Sliding window configuration
WINDOW_SIZE = 2      # Number of intervals (time points) in window
WINDOW_STEP = 1      # Step size in intervals

# Animation settings
FRAME_DURATION_MS = 800  # Milliseconds per frame

# Output file
OUTPUT_FILE = f'2H_wear_animation_{CHANNEL}.gif'

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

def build_matrix(dfs, hours_list):
    """Build frequency x time matrix of 2H values."""
    freqs = dfs[hours_list[0]]['Frequency'].values
    matrix = []
    for freq in freqs:
        row = []
        for h in hours_list:
            if h in dfs:
                v = dfs[h][dfs[h]['Frequency'] == freq]['2nd Harmonic'].values
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
        
        window_stats.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_hours': hours_list[start_idx],
            'end_hours': hours_list[end_idx],
            'pct_vals': pct_vals,
            'mean': mean_val
        })
    
    return window_stats

# =============================================================================
# ANIMATION
# =============================================================================

def create_animation(hours_list, matrix, freqs, window_size, window_step, output_file, frame_duration, channel):
    """Create animated GIF with sliding window."""
    
    # Precompute all window stats
    window_stats = precompute_all_windows(matrix, hours_list, window_size, window_step)
    n_windows = len(window_stats)
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid: top row has heatmap and bar chart (2 units), bottom row has summary (2 units)
    ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=2)  # Heatmap
    ax2 = plt.subplot2grid((4, 2), (0, 1), rowspan=2)  # Bar chart
    ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=2, rowspan=2)  # Summary line chart (twice as tall)
    
    fig.suptitle(f'{channel} CHANNEL - Sliding Window Transition Analysis', fontsize=14, fontweight='bold')
    
    # Panel 1: Static heatmap
    im = ax1.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=-50, vmax=-20)
    ax1.set_xticks(range(len(hours_list)))
    ax1.set_xticklabels(hours_list)
    tick_spacing = max(1, len(freqs) // 10)
    ax1.set_yticks(range(0, len(freqs), tick_spacing))
    ax1.set_yticklabels([f'{int(freqs[i])}' for i in range(0, len(freqs), tick_spacing)])
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('2H Distortion Heatmap')
    plt.colorbar(im, ax=ax1, label='2H (dB)')
    
    # Window highlight rectangle
    rect = patches.Rectangle((0, 0), window_size, len(freqs), 
                              linewidth=3, edgecolor='white', facecolor='none', linestyle='-')
    ax1.add_patch(rect)
    
    # Panel 2: Transition bar chart
    bars = ax2.bar(range(len(freqs)), [50]*len(freqs), width=1.0, color='steelblue', alpha=0.7)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='Random (50%)')
    mean_line = ax2.axhline(y=50, color='green', linestyle='-', linewidth=2, label='Mean')
    
    ax2.set_xticks(range(0, len(freqs), tick_spacing))
    ax2.set_xticklabels([f'{int(freqs[i])}' for i in range(0, len(freqs), tick_spacing)], rotation=45)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('% Transitions Increasing')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper right', fontsize=9)
    window_title = ax2.set_title('Window: T000-T048')
    
    # Panel 3: Summary line chart (builds up over time)
    window_labels = [f"T{ws['start_hours']:03d}-\nT{ws['end_hours']:03d}" for ws in window_stats]
    all_means = [ws['mean'] for ws in window_stats]
    
    ax3.set_xlim(-0.5, n_windows - 0.5)
    ax3.set_ylim(0, 100)
    ax3.set_xticks(range(n_windows))
    ax3.set_xticklabels(window_labels, fontsize=9)
    ax3.set_xlabel('Window')
    ax3.set_ylabel('Mean % Increasing')
    ax3.set_title('Summary: Mean Transition % by Window')
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Random (50%)')
    ax3.grid(True, alpha=0.3)
    
    # Initialize summary elements
    summary_line, = ax3.plot([], [], 'o-', color='darkblue', linewidth=2, markersize=8)
    summary_bars = ax3.bar(range(n_windows), [0]*n_windows, width=0.6, color='steelblue', alpha=0.5)
    current_marker, = ax3.plot([], [], 'o', color='lime', markersize=15, markeredgecolor='black', markeredgewidth=2)
    ax3.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    def update(frame):
        ws = window_stats[frame]
        
        # Update rectangle position
        rect.set_xy((ws['start_idx'] - 0.5, -0.5))
        rect.set_width(window_size)
        rect.set_height(len(freqs))
        
        # Update bars
        for bar, pct in zip(bars, ws['pct_vals']):
            bar.set_height(pct)
        
        # Update mean line
        mean_line.set_ydata([ws['mean'], ws['mean']])
        
        # Update title
        window_title.set_text(f"Window: T{ws['start_hours']:03d}-T{ws['end_hours']:03d} (Mean: {ws['mean']:.1f}%)")
        
        # Update summary - show all bars up to current frame
        for i, bar in enumerate(summary_bars):
            if i <= frame:
                bar.set_height(all_means[i])
                # Color code: green if >60%, red if <50%, yellow otherwise
                if all_means[i] >= 60:
                    bar.set_color('forestgreen')
                elif all_means[i] < 50:
                    bar.set_color('indianred')
                else:
                    bar.set_color('gold')
            else:
                bar.set_height(0)
        
        # Update summary line
        x_data = list(range(frame + 1))
        y_data = all_means[:frame + 1]
        summary_line.set_data(x_data, y_data)
        
        # Update current marker
        current_marker.set_data([frame], [ws['mean']])
        
        return [rect] + list(bars) + [mean_line, window_title, summary_line, current_marker] + list(summary_bars)
    
    anim = FuncAnimation(fig, update, frames=n_windows, interval=frame_duration, blit=False)
    
    writer = PillowWriter(fps=1000/frame_duration)
    anim.save(output_file, writer=writer)
    print(f"Saved animation to {output_file}")
    
    # Print summary table
    print(f"\nWindow Summary:")
    print(f"{'Window':<15} {'Mean %':<10} {'Signal'}")
    print("-" * 35)
    for ws in window_stats:
        signal = "CLEAR" if ws['mean'] >= 60 else ("noise" if ws['mean'] >= 45 else "inverse?")
        print(f"T{ws['start_hours']:03d}-T{ws['end_hours']:03d}     {ws['mean']:5.1f}%     {signal}")
    
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    dfs = load_data(HOURS, CHANNEL, FILE_PATTERN, FREQ_MIN, FREQ_MAX)
    
    if not dfs:
        print("No data loaded!")
        return
    
    matrix, freqs = build_matrix(dfs, HOURS)
    
    print(f"Data: {len(freqs)} frequencies, {len(HOURS)} time points")
    print(f"Window: {WINDOW_SIZE} intervals, step {WINDOW_STEP}")
    print(f"Generating animation...")
    
    create_animation(HOURS, matrix, freqs, WINDOW_SIZE, WINDOW_STEP, OUTPUT_FILE, FRAME_DURATION_MS, CHANNEL)

if __name__ == '__main__':
    main()
