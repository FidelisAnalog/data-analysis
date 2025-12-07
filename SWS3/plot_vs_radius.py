#!/usr/bin/env python3
"""
Stylus Wear Analysis - Harmonic Distortion vs Contact Geometry
Plots 2H and 3H distortion against minor radius to show geometry effects.
Requires: pip install adjustText
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

# =============================================================================
# CONFIGURATION
# =============================================================================

# Channel to analyze: 'L' or 'R'
CHANNEL = 'L'

# File pattern
FILE_PATTERN = './data/SWS3_VM95ML_REV_T{hours:03d}_{channel}.csv'

# Hours to analyze
HOURS = [0, 48, 96, 192, 240, 288, 384, 480, 576, 672]

# Minor radius progression (mils) - update if analyzing different channel
# L channel values shown; R channel would use R Minor from contact patch data
MINOR_RADIUS = {
    'L': {0: 0.08, 48: 0.13, 96: 0.21, 192: 0.24, 240: 0.24, 288: 0.26, 
          384: 0.32, 480: 0.33, 576: 0.33, 672: 0.33},
    'R': {0: 0.14, 48: 0.17, 96: 0.20, 192: 0.26, 240: 0.27, 288: 0.29,
          384: 0.33, 480: 0.34, 576: 0.35, 672: 0.35},
}

# Frequencies for 6-panel 2H vs 3H comparison
COMPARISON_FREQS = [5500, 5700, 5900, 6000, 6200, 6500]

# Frequencies for even/odd balance plot
BALANCE_FREQS = [5900, 6000, 6200]

# Output files
OUTPUT_COMPARISON = './outputs/distortion_vs_radius_{channel}.png'
OUTPUT_BALANCE = './outputs/even_odd_balance_vs_radius_{channel}.png'

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(hours_list, channel, file_pattern):
    """Load all CSV files."""
    dfs = {}
    for h in hours_list:
        filepath = file_pattern.format(hours=h, channel=channel)
        try:
            df = pd.read_csv(filepath)
            dfs[h] = df
        except FileNotFoundError:
            print(f"Warning: File not found for T{h:03d}")
    return dfs

# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(dfs, hours_list, minor_radius, freqs, channel, output_file):
    """Create 6-panel 2H vs 3H comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f'{channel} Channel: 2H vs 3H Distortion vs Minor Radius', fontsize=14, fontweight='bold')
    
    for idx, freq in enumerate(freqs):
        ax = axes[idx // 3, idx % 3]
        
        radii = []
        h2_vals = []
        h3_vals = []
        
        for h in hours_list:
            r = minor_radius[h]
            v2 = dfs[h][dfs[h]['Frequency'] == freq]['2nd Harmonic'].values
            v3 = dfs[h][dfs[h]['Frequency'] == freq]['3rd Harmonic'].values
            
            if len(v2) > 0 and len(v3) > 0:
                radii.append(r)
                h2_vals.append(v2[0])
                h3_vals.append(v3[0])
        
        ax.plot(radii, h2_vals, 'o-', color='blue', label='2H', linewidth=1.5, markersize=6)
        ax.plot(radii, h3_vals, 's-', color='red', label='3H', linewidth=1.5, markersize=6)
        
        # Collect all annotations for adjustText
        texts = []
        for i, h in enumerate(hours_list):
            txt = ax.text(radii[i], h2_vals[i], f'T{h}', fontsize=7, color='blue', alpha=0.9)
            texts.append(txt)
            txt = ax.text(radii[i], h3_vals[i], f'T{h}', fontsize=7, color='red', alpha=0.9)
            texts.append(txt)
        
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5),
                    expand_points=(1.5, 1.5), force_text=(0.5, 0.5))
        
        ax.set_xlabel('Minor Radius (mils)')
        ax.set_ylabel('Distortion (dB)')
        ax.set_title(f'{freq} Hz')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-60, -20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved {output_file}")

def plot_balance(dfs, hours_list, minor_radius, freqs, channel, output_file):
    """Create even/odd balance plot."""
    fig, ax_main = plt.subplots(figsize=(12, 7))
    fig.suptitle(f'{channel} Channel: 2H minus 3H (Even/Odd Balance)', fontsize=14, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    for fidx, freq in enumerate(freqs):
        radii = []
        diff_vals = []
        
        for h in hours_list:
            r = minor_radius[h]
            v2 = dfs[h][dfs[h]['Frequency'] == freq]['2nd Harmonic'].values
            v3 = dfs[h][dfs[h]['Frequency'] == freq]['3rd Harmonic'].values
            
            if len(v2) > 0 and len(v3) > 0:
                radii.append(r)
                diff_vals.append(v2[0] - v3[0])
        
        ax_main.plot(radii, diff_vals, marker=markers[fidx], linestyle='-', color=colors[fidx],
                     label=f'{freq} Hz', linewidth=1.5, markersize=8)
    
    # Annotate middle frequency points
    mid_freq = freqs[len(freqs) // 2]
    texts = []
    for h in hours_list:
        r = minor_radius[h]
        v2 = dfs[h][dfs[h]['Frequency'] == mid_freq]['2nd Harmonic'].values
        v3 = dfs[h][dfs[h]['Frequency'] == mid_freq]['3rd Harmonic'].values
        if len(v2) > 0 and len(v3) > 0:
            diff = v2[0] - v3[0]
            # Bold labels for clustered points
            is_clustered = list(minor_radius.values()).count(r) > 1
            txt = ax_main.text(r, diff, f'T{h}', fontsize=9, 
                              fontweight='bold' if is_clustered else 'normal')
            texts.append(txt)
    
    adjust_text(texts, ax=ax_main, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5),
                expand_points=(1.5, 1.5), force_text=(0.5, 0.5))
    
    ax_main.set_xlabel('Minor Radius (mils)', fontsize=12)
    ax_main.set_ylabel('2H minus 3H (dB)', fontsize=12)
    ax_main.legend(loc='lower right', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add vertical lines at clustered radii
    seen = set()
    for r in minor_radius.values():
        if list(minor_radius.values()).count(r) > 1 and r not in seen:
            ax_main.axvline(x=r, color='gray', linestyle=':', alpha=0.5)
            seen.add(r)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved {output_file}")

def print_table(dfs, hours_list, minor_radius, freq):
    """Print data table for a specific frequency."""
    print("\n" + "=" * 70)
    print(f"2H minus 3H (Even/Odd Balance) at {freq} Hz")
    print("=" * 70)
    print(f"{'Hours':<8} {'Radius':<10} {'2H':<10} {'3H':<10} {'2H minus 3H':<12}")
    print("-" * 52)
    for h in hours_list:
        r = minor_radius[h]
        v2 = dfs[h][dfs[h]['Frequency'] == freq]['2nd Harmonic'].values[0]
        v3 = dfs[h][dfs[h]['Frequency'] == freq]['3rd Harmonic'].values[0]
        print(f"T{h:<7} {r:<10.2f} {v2:<10.1f} {v3:<10.1f} {v2-v3:<+12.1f}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    minor_radius = MINOR_RADIUS[CHANNEL]
    
    dfs = load_data(HOURS, CHANNEL, FILE_PATTERN)
    
    if not dfs:
        print("No data loaded!")
        return
    
    output_comparison = OUTPUT_COMPARISON.format(channel=CHANNEL)
    output_balance = OUTPUT_BALANCE.format(channel=CHANNEL)
    
    plot_comparison(dfs, HOURS, minor_radius, COMPARISON_FREQS, CHANNEL, output_comparison)
    plot_balance(dfs, HOURS, minor_radius, BALANCE_FREQS, CHANNEL, output_balance)
    print_table(dfs, HOURS, minor_radius, 6000)

if __name__ == '__main__':
    main()
