# Stylus Wear Analysis - AI Onboarding Document

This document provides context for AI assistants working with this phono cartridge wear analysis dataset. Read this before diving into the data or scripts.

---

## Project Overview

**Hypothesis:** As a phono cartridge stylus wears, its contact patch grows larger, degrading high-frequency tracing ability and increasing harmonic distortion.

**Cartridge:** Audio-Technica VM95ML with MicroRidge stylus (Moving Magnet type)

**Test Method:** Constant velocity log sweep test record measured at 10 intervals from 0 to 672 hours of use. Between measurements, the stylus was used for normal music playback to accumulate wear.

---

## Data Files

### Frequency Response CSVs
- Pattern: `SWS3_VM95ML_REV_T{hours:03d}_{channel}.csv`
- Hours: 000, 048, 096, 192, 240, 288, 384, 480, 576, 672
- Channels: L (left), R (right)
- Columns: `Frequency`, `Amplitude`, `Crosstalk`, `2nd Harmonic`, `3rd Harmonic`
- All harmonic values are in dB (negative values; closer to 0 = more distortion)

### Contact Patch Measurements
Stylus geometry was measured at each interval. Key dimensions (in mils):

| Hours | L Minor | R Minor | L Major | R Major | Tip Span |
|-------|---------|---------|---------|---------|----------|
| 0     | 0.08    | 0.14    | 1.01    | 0.97    | 1.20     |
| 48    | 0.13    | 0.17    | 1.16    | 1.02    | 1.02     |
| 96    | 0.21    | 0.20    | 1.17    | 1.18    | 0.88     |
| 192   | 0.24    | 0.26    | 1.30    | 1.20    | 0.67     |
| 240   | 0.24    | 0.27    | 1.31    | 1.22    | 0.62     |
| 288   | 0.26    | 0.29    | 1.34    | 1.31    | 0.53     |
| 384   | 0.32    | 0.33    | 1.35    | 1.34    | 0.48     |
| 480   | 0.33    | 0.34    | 1.36    | 1.38    | 0.47     |
| 576   | 0.33    | 0.35    | 1.42    | 1.39    | 0.30     |
| 672   | 0.33    | 0.35    | 1.42    | 1.39    | 0.29     |

**Terminology note:** "Minor radius" is the tracing-critical dimension (narrow edge of MicroRidge contact patch). "Major radius" is the longer dimension along the groove wall.

---

## Critical Technical Context

### MM Cartridge Bandwidth Limitations
Moving Magnet cartridges have high inductance causing electrical rolloff above ~15-20 kHz. This affects harmonic measurement:
- 2H of 10 kHz fundamental = 20 kHz harmonic (edge of bandwidth)
- 2H of 12 kHz fundamental = 24 kHz harmonic (attenuated/unreliable)
- 3H of 6.5 kHz fundamental = 19.5 kHz harmonic (edge of bandwidth)

**Practical limits:**
- 2H analysis: fundamentals up to ~10 kHz are reliable
- 3H analysis: fundamentals up to ~6.5 kHz are reliable

### MicroRidge Stylus Geometry
The MicroRidge has an asymmetric contact patch — narrow in the direction of groove travel, longer along the groove wall. The L and R groove walls contact different faces of the stylus, so wear can be asymmetric between channels.

### Constant Velocity Test Record
The test record is cut at constant velocity, meaning groove amplitude is inversely proportional to frequency. The cartridge is a velocity-sensitive transducer, so its output is nominally flat. No RIAA equalization is involved.

### SRA Error
The stylus rake angle was approximately 4° off from optimal (92°). This fixed angular error has more impact as the minor radius grows — a larger contact patch means more of the stylus face interacts with the groove at the "wrong" angle.

---

## Key Findings

### Two Distinct Wear Regimes

**Break-in period (T000-T192):**
- Rapid contact patch growth
- Clear, measurable increase in 2H distortion
- 85-97% of frequency points show increasing distortion step-to-step
- This is real signal

**Post-break-in (T192-T576):**
- Contact dimensions stabilize
- Step-to-step transitions drop to ~50% (coin flip)
- Linear regression still shows positive slope due to endpoint separation
- Individual intervals are dominated by measurement noise

### Channel Asymmetry
L and R channels show different frequency-dependent wear signatures:
- L channel: strongest 2H trend at 1-6 kHz
- R channel: strongest 2H trend at 6-8 kHz

This may relate to MicroRidge asymmetry and different wear patterns on each groove wall contact face.

### T576-T672 Anomaly
Contact patch dimensions stopped changing after T576, but L channel 2H jumped significantly at T672. The tester observed evidence of stylus bottoming (vinyl dust) during the music playback that accumulated hours for T672. This represents a different failure mode (mechanical impact) rather than gradual tracing wear.

### T480 3H Spike
At 6 kHz, 3rd harmonic distortion spiked dramatically at T480 (and partially T576) while 2H moved in the opposite direction. The 2H-minus-3H gap collapsed from +23 dB to +5 dB at identical contact geometry (all three points at 0.33 mil minor radius). This indicates a measurement setup variable (VTF, antiskate, azimuth) rather than geometry change. Even vs odd harmonic balance is sensitive to groove contact symmetry.

### 3H Shows No Wear Trend
Unlike 2H, the 3rd harmonic shows essentially no correlation with wear:
- Overall slope: -0.10 dB/100h (essentially flat)
- R² = 0.06 (no correlation)

The wear signature is primarily in even-order distortion.

---

## Analysis Scripts

### stylus_wear_analysis.py
Static 4-panel visualization per channel:
1. Individual frequency traces over time
2. Band averages with error bars
3. Heatmap (frequency vs time)
4. Step-to-step transition consistency

**Key config variables:**
- `HARMONIC`: '2H' or '3H'
- `CHANNEL`: 'L' or 'R'
- `TRANSITION_FREQ_MIN/MAX`: Constrain the transition analysis frequency range
- `TRANSITION_HOURS_MIN/MAX`: Constrain to specific time periods

### stylus_wear_animation.py
Animated GIF showing sliding window analysis:
- Heatmap with moving window highlight
- Per-frequency transition percentages for current window
- Summary bar chart building up over time

**Key config variables:**
- `HARMONIC`: '2H' or '3H'
- `CHANNEL`: 'L' or 'R'
- `WINDOW_SIZE`: Number of intervals per window (default 2)
- `FREQ_MIN/MAX`: Frequency range for analysis

### plot_vs_radius.py
Plots distortion against contact patch minor radius rather than hours:
- 6-panel 2H vs 3H comparison at specific frequencies
- Even/odd balance plot (2H minus 3H)

Useful for seeing geometry relationships. Points at same x-coordinate have identical measured contact dimensions — any vertical spread is due to non-geometry factors.

---

## Lessons Learned / Gotchas

1. **Don't trust linear regression blindly.** The full-dataset R² looks good, but step-to-step analysis reveals the trend is driven by endpoints with noise in between.

2. **Constrain your analysis range.** Below 1 kHz, tracing geometry is less relevant. Above 10 kHz (2H) or 6.5 kHz (3H), MM electrical rolloff corrupts harmonic measurements.

3. **Watch for clustered x-values.** T192/T240 some channels have identical minor radius (0.24). T480/T576/T672 0.33. Plotting against radius reveals whether differences are geometry-driven or setup-driven.

4. **Check both harmonics.** 2H and 3H can move in opposite directions.

5. **The T480 interval has anomalies.** 3H spikes at 6 kHz. Factor this into any analysis of that region.

6. **T672 may include bottoming damage.** Contact dimensions stopped changing but distortion jumped. Different failure mode.

7. **L and R are not interchangeable.** They show different frequency-dependent patterns.

---

## Suggested Starting Points

**To reproduce the core analysis:**
```python
# Set CHANNEL = 'L', HARMONIC = '2H' in stylus_wear_analysis.py
# Run with TRANSITION_HOURS_MIN = None to see full dataset
# Then set TRANSITION_HOURS_MIN = 192 to see post-break-in noise
```

**To see the break-in vs mature wear distinction:**
```python
# Run stylus_wear_animation.py with HARMONIC = '2H'
# Watch the summary bar chart — green bars cluster at T048-T192
```

**To investigate the T480 anomaly:**
```python
# Run plot_vs_radius.py
# Look at 6000 Hz panel — T480 3H (red) is way above T672
```

---

## Questions This Analysis Can't Answer

- Whether the wear pattern is specific to MicroRidge geometry or generalizes to other stylus profiles
- How much of the measurement variance is stylus vs test record condition vs setup repeatability
- What the "acceptable" distortion threshold is for audibility
- Whether the SRA error is a significant confound
- Identify spurious inputs or other confounds such as temperature, humidity, seismic, setup 

---

*Document generated from analysis session. Data and samples collected by original tester; analysis performed collaboratively with AI assistance.*
