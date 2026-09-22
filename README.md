# Yongsan Station BLE Real-Time Indoor Tracker

A single-file Python tool that estimates and visualizes a person's real-time indoor position at **Yongsan Station** using RSSI readings from a set of fixed BLE beacons ("brain" series), with a live, interactive `matplotlib` UI for beacon placement, calibration, and position playback.

## What it does

- Scans nearby BLE advertisements with [`bleak`](https://github.com/hbldh/bleak) and matches known beacon MAC addresses.
- Converts each beacon's RSSI to an estimated distance (log-distance path-loss model).
- Estimates the most likely (x, y) position by minimizing a weighted least-squares cost over a **0.1 m resolution grid** covering the building footprint, rather than solving trilateration analytically — this stays stable even when 1–2 beacons are active.
- Renders the building outline, beacon positions, live position marker, and a movement trail on an interactive `matplotlib` figure, with on-screen buttons to add/remove/move beacons, calibrate, save, and reset.

## Positioning pipeline

1. **Signal smoothing (dual filter)** — a rolling median (last 5 samples) removes spikes, then an asymmetric exponential moving average is applied: it reacts quickly when the signal gets *stronger* (weight 0.5) but smooths heavily when it gets *weaker* (weight 0.05), so the estimate doesn't lag behind real approach but doesn't jitter on noisy drop-offs.
2. **RSSI → distance** — a standard log-distance path-loss formula (`tx`, path-loss exponent `n = 3.0`), with a **30% signal-boost ratio** applied to compensate for measured signal attenuation at this specific site, clamped to a 0.3 m minimum.
3. **Position estimation (heatmap / grid search)** — for every point on a 0.1 m grid over the building, a weighted squared-error cost against all active beacons' estimated distances is computed; the grid point with minimum cost is the estimated position. Beacons with a shorter estimated distance are weighted more heavily (`1 / (d³ + 0.1)`).
4. **Velocity clamp / temporal smoothing** — the new raw estimate is blended with the previous position using a distance-adaptive weight: small movements (<0.5 m) are heavily smoothed (10% new / 90% old) to suppress standing-still jitter, large jumps (>5 m) are treated as likely noise and barely trusted (5% new), and normal walking-speed movement is blended at 40% new / 60% old.
5. **Building clamp** — the final position is clamped to the building's rectangular bounds so estimates can't drift outside the physical space.
6. **Beacon dropout handling** — a beacon is ignored if not seen within the last 1.5 seconds (`SEEN_TIMEOUT`) or if its filtered RSSI falls below a cutoff (`RSSI_CUTOFF = -88 dBm`).

## Interactive UI

Built with `matplotlib.widgets.Button` and `FuncAnimation` (10 fps refresh):

| Control | Action |
|---|---|
| Drag a beacon marker | Reposition it on the floor plan (snapped to a 0.5 m grid) |
| **추가** (Add) | Activate a spare beacon slot (up to 10 active beacons) |
| **삭제** (Delete) | Toggle delete mode, then click a beacon to deactivate it (minimum 3 active) |
| **1m보정** (1 m calibrate) | Set the selected beacon's `tx` (RSSI @ 1 m) from its current live reading |
| **저장** (Save) | Persist beacon layout to `beacon_positions.json` |
| **초기화** (Reset) | Restore the default beacon layout |
| Keyboard shortcuts | `a` add, `d` delete selected, `c` calibrate, `s` save, `r` reset |

Beacon layout (name, position, and active state) is persisted to `beacon_positions.json` and reloaded on startup if present.

## Requirements

```
python >= 3.9
numpy
matplotlib
bleak        # optional at import time, required to actually scan BLE hardware
```

Install with:

```bash
pip install numpy matplotlib bleak
```

## Usage

```bash
python yongsan_ble_realtime_tracker.py
```

This opens the live tracker window and starts a background BLE scan thread. If `bleak` is not installed or no Bluetooth adapter is available, the UI still opens but no live position will be computed until beacons are seen.

## Notes

- Beacon MAC addresses (`BEACON_MACS`) and the default 11-beacon layout (`DEFAULT_LAYOUT`) are hardcoded for the Yongsan Station test deployment and will need to be updated for a different site.
- Building dimensions, the internal pillar, and the exit location are defined in the `BUILDING` dict and drawn as reference geometry on the plot.
