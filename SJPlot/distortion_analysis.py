#!/usr/bin/env python3
"""
2nd Harmonic Distortion vs Hours
Analyzes phono cartridge wear by tracking 2H distortion over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# CONFIGURATION
# =============================================================================

# Channel to analyze: 'L' or 'R'
CHANNEL = 'L'

# File pattern - adjust path and naming as needed
# Use {hours:03d} for zero-padded hours, {channel} for L/R
FILE_PATTERN = 'SWS3_VM95ML_REV_T{hours:03d}_{channel}.csv'

# Hours to analyze - add/remove as data becomes available
HOURS = [0, 48, 96, 192, 240, 288, 384, 480, 576, 672]

# Frequency range for main analysis (Hz)
MIN_FREQ = 1000

# Frequency bands for band analysis (Hz)
BANDS = [(1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000), (10000, 20000)]

# Transition plot constraints
# Set to None to use full range
TRANSITION_FREQ_MIN = 2000    # Hz, or None for MIN_FREQ
TRANSITION_FREQ_MAX = 9000    # Hz, or None for no upper limit
TRANSITION_HOURS_MIN = 192    # Hours, or None for first measurement
TRANSITION_HOURS_MAX = None   # Hours, or None for last measurement

# Output file
OUTPUT_FILE = f'2H_wear_analysis_{CHANNEL}.png'

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(hours_list, channel, file_pattern, min_freq):
    """Load all CSV files and filter to frequency range."""
    dfs = {}
    for h in hours_list:
        filepath = file_pattern.format(hours=h, channel=channel)
        try:
            df = pd.read_csv(filepath)
            df = df[df['Frequency'] >= min_freq].copy()
            dfs[h] = df
        except FileNotFoundError:
            print(f"Warning: File not found for T{h:03d}")
    return dfs

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_overall_stats(dfs, hours_list):
    """Calculate overall mean 2H and linear regression."""
    means = []
    for h in hours_list:
        if h in dfs:
            means.append(dfs[h]['2nd Harmonic'].mean())
        else:
            means.append(np.nan)
    
    valid = [(hours_list[i], means[i]) for i in range(len(means)) if not np.isnan(means[i])]
    x, y = zip(*valid)
    
    coeffs = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    
    return {
        'hours': hours_list,
        'means': means,
        'slope_per_100h': coeffs[0] * 100,
        'r2': r2
    }

def calculate_band_stats(dfs, hours_list, bands):
    """Calculate statistics by frequency band."""
    results = {}
    for lo, hi in bands:
        if hi >= 20000:
            band_name = f"{lo//1000}k+"
        else:
            band_name = f"{lo//1000}-{hi//1000}k"
        band_means = []
        band_stds = []
        
        for h in hours_list:
            if h in dfs:
                band_data = dfs[h][(dfs[h]['Frequency'] >= lo) & (dfs[h]['Frequency'] < hi)]['2nd Harmonic']
                band_means.append(band_data.mean())
                band_stds.append(band_data.std())
            else:
                band_means.append(np.nan)
                band_stds.append(np.nan)
        
        valid = [(hours_list[i], band_means[i]) for i in range(len(band_means)) if not np.isnan(band_means[i])]
        if len(valid) > 2:
            x, y = zip(*valid)
            slope = np.polyfit(x, y, 1)[0]
            r2 = np.corrcoef(x, y)[0, 1] ** 2
        else:
            slope = np.nan
            r2 = np.nan
        
        results[band_name] = {
            'means': band_means,
            'stds': band_stds,
            'slope_per_100h': slope * 100,
            'r2': r2
        }
    
    return results

def calculate_monotonicity(dfs, hours_list, freq_min=None, freq_max=None, hours_min=None, hours_max=None):
    """Calculate step-to-step monotonicity for each frequency."""
    if not dfs:
        return [], []
    
    # Filter hours
    filtered_hours = hours_list.copy()
    if hours_min is not None:
        filtered_hours = [h for h in filtered_hours if h >= hours_min]
    if hours_max is not None:
        filtered_hours = [h for h in filtered_hours if h <= hours_max]
    
    # Get frequencies
    common_freqs = dfs[hours_list[0]]['Frequency'].values
    
    # Filter frequencies
    if freq_min is not None:
        common_freqs = common_freqs[common_freqs >= freq_min]
    if freq_max is not None:
        common_freqs = common_freqs[common_freqs < freq_max]
    
    results = []
    
    for freq in common_freqs:
        vals = []
        for h in filtered_hours:
            if h in dfs:
                v = dfs[h][dfs[h]['Frequency'] == freq]['2nd Harmonic'].values
                vals.append(v[0] if len(v) > 0 else np.nan)
            else:
                vals.append(np.nan)
        
        vals = np.array(vals)
        
        n_increasing = 0
        n_transitions = 0
        
        for i in range(len(vals) - 1):
            if not np.isnan(vals[i]) and not np.isnan(vals[i + 1]):
                n_transitions += 1
                if vals[i + 1] > vals[i]:
                    n_increasing += 1
        
        valid_vals = vals[~np.isnan(vals)]
        is_monotonic = all(valid_vals[i] <= valid_vals[i + 1] for i in range(len(valid_vals) - 1)) if len(valid_vals) > 1 else False
        
        results.append({
            'freq': freq,
            'vals': vals,
            'n_inc': n_increasing,
            'n_trans': n_transitions,
            'pct_inc': 100 * n_increasing / n_transitions if n_transitions > 0 else 0,
            'monotonic': is_monotonic
        })
    
    return results, filtered_hours

# =============================================================================
# PRINT ANALYSIS
# =============================================================================

def print_analysis(channel, hours_list, overall, band_stats, mono_results, mono_hours, constraints):
    """Print analysis summary to console."""
    print("=" * 70)
    print(f"{channel} CHANNEL: 2nd HARMONIC DISTORTION ANALYSIS")
    print("=" * 70)
    
    print("\n--- Mean 2nd Harmonic (dB) vs Hours ---")
    for i, h in enumerate(hours_list):
        if not np.isnan(overall['means'][i]):
            print(f"T{h:03d}: {overall['means'][i]:.2f} dB")
    
    print(f"\nLinear fit: slope = {overall['slope_per_100h']:.4f} dB per 100 hours")
    print(f"            R² = {overall['r2']:.3f}")
    
    print("\n--- Band Analysis ---")
    for band_name, stats in band_stats.items():
        print(f"{band_name}: slope = {stats['slope_per_100h']:+.3f} dB/100h, R² = {stats['r2']:.3f}")
    
    print("\n--- Monotonicity Analysis ---")
    freq_min, freq_max, hours_min, hours_max = constraints
    constraint_str = f"{freq_min or 'min'}-{freq_max or 'max'} Hz, T{hours_min or hours_list[0]}-T{hours_max or hours_list[-1]}"
    print(f"Constraints: {constraint_str}")
    print(f"Hours analyzed: {mono_hours}")
    
    n_mono = sum(1 for r in mono_results if r['monotonic'])
    pct_vals = [r['pct_inc'] for r in mono_results]
    
    print(f"Total frequency points: {len(mono_results)}")
    print(f"Strictly monotonic: {n_mono} ({100*n_mono/len(mono_results):.1f}%)")
    print(f"Mean % increasing transitions: {np.mean(pct_vals):.1f}%")
    print(f"Median: {np.median(pct_vals):.1f}%")
    print(f"Range: {np.min(pct_vals):.1f}% - {np.max(pct_vals):.1f}%")

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(channel, hours_list, dfs, band_stats, mono_results, mono_hours, output_file, constraints):
    """Create 4-panel visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{channel} CHANNEL - 2H Distortion vs Stylus Wear', fontsize=14, fontweight='bold')
    
    freq_min, freq_max, hours_min, hours_max = constraints
    
    # Panel 1: Individual frequency traces
    ax1 = axes[0, 0]
    selected_freqs = [1000, 3000, 5000, 7000, 8000, 10000]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(selected_freqs)))
    
    for freq, color in zip(selected_freqs, colors):
        vals = []
        for h in hours_list:
            if h in dfs:
                v = dfs[h][dfs[h]['Frequency'] == freq]['2nd Harmonic'].values
                vals.append(v[0] if len(v) > 0 else np.nan)
            else:
                vals.append(np.nan)
        ax1.plot(hours_list, vals, 'o-', color=color, label=f'{freq} Hz', linewidth=1.5, markersize=5)
    
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('2nd Harmonic (dB)')
    ax1.set_title('2H Distortion vs Wear Time - Selected Frequencies')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-60, -20)
    
    # Panel 2: Band averages with error bars
    ax2 = axes[0, 1]
    band_colors = ['navy', 'darkgreen', 'darkred', 'purple', 'darkorange', 'gray']
    
    for (band_name, stats), color in zip(band_stats.items(), band_colors):
        ax2.errorbar(hours_list, stats['means'], yerr=stats['stds'], fmt='o-', color=color,
                     label=band_name, linewidth=1.5, markersize=5, capsize=3)
    
    ax2.set_xlabel('Hours')
    ax2.set_ylabel('2nd Harmonic (dB)')
    ax2.set_title('2H by Frequency Band (mean ± std dev)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-60, -20)
    
    # Panel 3: Heatmap
    ax3 = axes[1, 0]
    common_freqs = dfs[hours_list[0]]['Frequency'].values
    freq_mask = common_freqs <= 10000
    plot_freqs = common_freqs[freq_mask]
    
    matrix = []
    for freq in plot_freqs:
        row = []
        for h in hours_list:
            if h in dfs:
                v = dfs[h][dfs[h]['Frequency'] == freq]['2nd Harmonic'].values
                row.append(v[0] if len(v) > 0 else np.nan)
            else:
                row.append(np.nan)
        matrix.append(row)
    
    matrix = np.array(matrix)
    im = ax3.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=-50, vmax=-20)
    ax3.set_xticks(range(len(hours_list)))
    ax3.set_xticklabels(hours_list)
    ax3.set_yticks(range(0, len(plot_freqs), 10))
    ax3.set_yticklabels([f'{int(plot_freqs[i])}' for i in range(0, len(plot_freqs), 10)])
    ax3.set_xlabel('Hours')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('2H Distortion Heatmap (1-10 kHz)')
    plt.colorbar(im, ax=ax3, label='2H (dB)')
    
    # Draw selection rectangle on heatmap
    # Find indices for the selection area
    sel_freq_min = freq_min if freq_min is not None else plot_freqs[0]
    sel_freq_max = freq_max if freq_max is not None else plot_freqs[-1]
    sel_hours_min = hours_min if hours_min is not None else hours_list[0]
    sel_hours_max = hours_max if hours_max is not None else hours_list[-1]
    
    # Only draw if selection is within heatmap range
    if sel_freq_min < 10000 and sel_freq_max > plot_freqs[0]:
        # Find frequency indices
        freq_idx_min = np.searchsorted(plot_freqs, sel_freq_min)
        freq_idx_max = np.searchsorted(plot_freqs, min(sel_freq_max, 10000))
        
        # Find hour indices
        hour_idx_min = hours_list.index(sel_hours_min) if sel_hours_min in hours_list else 0
        hour_idx_max = hours_list.index(sel_hours_max) if sel_hours_max in hours_list else len(hours_list) - 1
        
        # Draw rectangle (x is hours, y is frequency)
        rect = patches.Rectangle(
            (hour_idx_min - 0.5, freq_idx_min - 0.5),
            hour_idx_max - hour_idx_min + 1,
            freq_idx_max - freq_idx_min,
            linewidth=2, edgecolor='white', facecolor='none', linestyle='--'
        )
        ax3.add_patch(rect)
    
    # Panel 4: Transition analysis (using constrained data)
    ax4 = axes[1, 1]
    pct_increasing = [r['pct_inc'] for r in mono_results]
    freqs = [r['freq'] for r in mono_results]
    
    ax4.bar(range(len(freqs)), pct_increasing, width=1.0, color='steelblue', alpha=0.7)
    ax4.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='Random (50%)')
    ax4.axhline(y=np.mean(pct_increasing), color='green', linestyle='-', linewidth=1.5,
                label=f'Mean ({np.mean(pct_increasing):.1f}%)')
    
    # Adaptive tick spacing based on number of frequencies
    tick_spacing = max(1, len(freqs) // 10)
    ax4.set_xticks(range(0, len(freqs), tick_spacing))
    ax4.set_xticklabels([f'{int(freqs[i])}' for i in range(0, len(freqs), tick_spacing)], rotation=45)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('% Transitions Increasing')
    
    # Build title with constraints
    constraint_parts = []
    if freq_min is not None or freq_max is not None:
        constraint_parts.append(f"{freq_min or 'min'}-{freq_max or 'max'} Hz")
    if hours_min is not None or hours_max is not None:
        constraint_parts.append(f"T{hours_min or hours_list[0]}+")
    constraint_str = f" [{', '.join(constraint_parts)}]" if constraint_parts else ""
    
    ax4.set_title(f'Step-to-Step Consistency{constraint_str}')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_file}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    dfs = load_data(HOURS, CHANNEL, FILE_PATTERN, MIN_FREQ)
    
    if not dfs:
        print("No data loaded!")
        return
    
    overall = calculate_overall_stats(dfs, HOURS)
    band_stats = calculate_band_stats(dfs, HOURS, BANDS)
    
    constraints = (TRANSITION_FREQ_MIN, TRANSITION_FREQ_MAX, TRANSITION_HOURS_MIN, TRANSITION_HOURS_MAX)
    mono_results, mono_hours = calculate_monotonicity(
        dfs, HOURS,
        freq_min=TRANSITION_FREQ_MIN,
        freq_max=TRANSITION_FREQ_MAX,
        hours_min=TRANSITION_HOURS_MIN,
        hours_max=TRANSITION_HOURS_MAX
    )
    
    print_analysis(CHANNEL, HOURS, overall, band_stats, mono_results, mono_hours, constraints)
    create_visualization(CHANNEL, HOURS, dfs, band_stats, mono_results, mono_hours, OUTPUT_FILE, constraints)

if __name__ == '__main__':
    main()
