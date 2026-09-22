"""
용산역 BLE 실시간 위치 측위 최종본 — 이중 필터, 속도 클램프, 0.1m 히트맵, 30% 신호 증폭 적용
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from bleak import BleakScanner
except ImportError:
    BleakScanner = None

import matplotlib
for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
    try:
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

BUILDING = dict(
    W_bottom=36.28, W_top=36.12, H=28.2,
    BL=(0.0, 0.0), BR=(36.28, 0.0), TL=(0.0, 28.2), TR=(36.12, 28.2),
    pillar=dict(x0=16.25, x1=18.54, depth=2.0),
    exit=(18.61, 36.12),
)
BUILDING_BOUNDS = (0.0, 0.0, 36.28, 28.2)

X_GRID, Y_GRID = np.meshgrid(np.arange(0, 36.3, 0.1), np.arange(0, 28.2, 0.1))

BEACON_MACS = {
    "brain0": "C3:00:00:5E:48:0D", "brain1": "C3:00:00:6A:5F:DE",
    "brain2": "C3:00:00:6A:5F:DF", "brain3": "C3:00:00:6A:5F:DD",
    "brain4": "C3:00:00:6A:5F:DC", "brain5": "C3:00:00:6A:5F:9D",
    "brain6": "C3:00:00:6A:5F:9C", "brain7": "C3:00:00:6A:5F:A7",
    "brain8": "C3:00:00:6A:5F:A6", "brain9": "C3:00:00:6A:5F:A2",
    "brain10": "C3:00:00:6A:5F:A3",
}

DEFAULT_LAYOUT = [
    ("brain0",  (0.0, 28.2),  "edge",   True),
    ("brain1",  (18.61, 28.2),"edge",   True),
    ("brain2",  (36.12, 28.2),"edge",   True),
    ("brain3",  (0.0, 0.0),   "edge",   True),
    ("brain4",  (17.40, 0.0), "edge",   True),
    ("brain5",  (36.28, 0.0), "edge",   True),
    ("brain6",  (14.0, 14.0), "center", True),
    ("brain7",  (22.0, 14.0), "center", True),
    ("brain8",  (18.0, 9.0),  "center", False),
    ("brain9",  (18.0, 19.0), "center", False),
    ("brain10", (18.0, 14.0), "center", False),
]

CONFIG_FILE = "beacon_positions.json"
MIN_ACTIVE, MAX_ACTIVE = 3, 10

TX_DEFAULT = -59.0
PATH_LOSS_N = 3.0                 
SEEN_TIMEOUT = 1.5                
RSSI_CUTOFF = -88.0               
DRAG_PICK_RADIUS = 1.8
SNAP = 0.5
SIGNAL_BOOST_RATIO = 1.3  

class Beacon:
    __slots__ = ("name", "mac", "xy", "kind", "tx", "active")
    def __init__(self, name, mac, xy, kind, tx, active):
        self.name = name; self.mac = mac.upper(); self.xy = [float(xy[0]), float(xy[1])]
        self.kind = kind; self.tx = float(tx); self.active = bool(active)

def rssi_to_distance(rssi, tx, n=PATH_LOSS_N):
    d = max(10 ** ((tx - rssi) / (10.0 * n)), 0.3)
    return d / SIGNAL_BOOST_RATIO  

def estimate_position_heatmap(points):
    if len(points) == 0:
        return None, "신호 없음(0개)"
    if len(points) == 1:
        return points[0][0].xy, "비콘1개(단일)"

    cost = np.zeros_like(X_GRID)
    for b, d in points:
        dist_to_grid = np.hypot(X_GRID - b.xy[0], Y_GRID - b.xy[1])
        weight = 1.0 / (d**3 + 0.1)
        cost += weight * (dist_to_grid - d)**2
        
    min_idx = np.unravel_index(np.argmin(cost), cost.shape)
    return (float(X_GRID[min_idx]), float(Y_GRID[min_idx])), f"히트맵({len(points)}개)"

def clamp_building(p):
    xm, ym, xM, yM = BUILDING_BOUNDS
    return (min(max(p[0], xm), xM), min(max(p[1], ym), yM))

class BLEReceiver:
    def __init__(self, mac_to_name: Dict[str, str]):
        self.mac_to_name = {m.upper(): n for m, n in mac_to_name.items()}
        self.name_set = set(mac_to_name.keys())
        self.lock = threading.Lock()
        self.latest: Dict[str, Tuple[float, float]] = {}
        self.error: Optional[str] = None
        self._thread = None

    def _on_adv(self, device, adv):
        mac = (device.address or "").upper()
        name = self.mac_to_name.get(mac)
        if name is None:
            adv_name = (getattr(adv, "local_name", None) or getattr(device, "name", None) or "")
            if adv_name in self.name_set:
                name = adv_name
        if name is not None:
            with self.lock:
                self.latest[name] = (adv.rssi, time.monotonic())

    async def _loop(self):
        scanner = BleakScanner(detection_callback=self._on_adv)
        await scanner.start()
        print("BLE 스캔 시작")
        try:
            while True:
                await __import__("asyncio").sleep(0.05)
        finally:
            await scanner.stop()

    def start(self):
        if BleakScanner is None:
            self.error = "bleak 미설치"
            return
        def run():
            import asyncio
            try: asyncio.run(self._loop())
            except Exception as e: self.error = str(e)
        self._thread = threading.Thread(target=run, daemon=True); self._thread.start()

    def snapshot(self):
        with self.lock:
            return dict(self.latest), self.error

class TrackerApp:
    def __init__(self, headless=False):
        self.beacons: List[Beacon] = self._load_or_default()
        self.buffers: Dict[str, List[float]] = {n: [] for n in BEACON_MACS}
        self.smoothed_rssi: Dict[str, float] = {}  
        
        self.last_pos: Optional[Tuple[float, float]] = None
        
        self.trail: List[Tuple[float, float]] = []
        self.selected: Optional[str] = None
        self.delete_mode = False
        self.drag_name: Optional[str] = None
        self.receiver = BLEReceiver(BEACON_MACS)
        self._build_figure(headless)

    def _default_beacons(self):
        return [Beacon(n, BEACON_MACS[n], xy, k, TX_DEFAULT, a) for n, xy, k, a in DEFAULT_LAYOUT]

    def _load_or_default(self):
        if os.path.exists(CONFIG_FILE):
            try:
                data = json.load(open(CONFIG_FILE, encoding="utf-8"))
                bs = [Beacon(d["name"], BEACON_MACS.get(d["name"], d.get("mac", "")),
                             d["xy"], d.get("kind", "center"), d.get("tx", TX_DEFAULT),
                             d.get("active", True)) for d in data]
                if bs: return bs
            except Exception: pass
        return self._default_beacons()

    def save(self, *_):
        data = [dict(name=b.name, mac=b.mac, xy=[round(b.xy[0], 2), round(b.xy[1], 2)],
                     kind=b.kind, tx=round(b.tx, 1), active=b.active) for b in self.beacons]
        json.dump(data, open(CONFIG_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        self._flash(f"저장됨 → {CONFIG_FILE}")

    def active(self): return [b for b in self.beacons if b.active]

    def add_beacon(self, *_):
        if len(self.active()) >= MAX_ACTIVE: return
        for b in self.beacons:
            if not b.active:
                b.active = True; self.selected = b.name
                self._rebuild_static(); return

    def remove_selected(self):
        if not self.selected or len(self.active()) <= MIN_ACTIVE: return
        for b in self.beacons:
            if b.name == self.selected and b.active:
                b.active = False; self.selected = None
                self._rebuild_static(); return

    def reset(self, *_):
        self.beacons = self._default_beacons(); self.selected = None
        self.last_pos = None; self.trail.clear(); self.smoothed_rssi.clear()
        for k in self.buffers: self.buffers[k].clear()
        self._rebuild_static()

    def calibrate_selected(self):
        if not self.selected: return
        latest, _ = self.receiver.snapshot(); now = time.monotonic()
        v = latest.get(self.selected)
        if not v or (now - v[1]) > SEEN_TIMEOUT: return
        for b in self.beacons:
            if b.name == self.selected:
                b.tx = round(v[0], 1); self._flash(f"{b.name} TX@1m = {b.tx}"); return

    def _compute(self):
        latest, err = self.receiver.snapshot()
        now = time.monotonic()
        
        points = []; info = {}
        
        for b in self.active():
            v = latest.get(b.name)
            if v is None or (now - v[1]) > SEEN_TIMEOUT:
                self.smoothed_rssi.pop(b.name, None)
                self.buffers[b.name].clear()
                continue
                
            raw_rssi = v[0]
            
            buf = self.buffers[b.name]
            buf.append(raw_rssi)
            if len(buf) > 5: buf.pop(0)
            median_rssi = float(np.median(buf))
            
            if b.name not in self.smoothed_rssi:
                self.smoothed_rssi[b.name] = median_rssi
            else:
                old_rssi = self.smoothed_rssi[b.name]
                if median_rssi > old_rssi:
                    self.smoothed_rssi[b.name] = old_rssi * 0.5 + median_rssi * 0.5
                else:
                    self.smoothed_rssi[b.name] = old_rssi * 0.95 + median_rssi * 0.05
                    
            filt_rssi = self.smoothed_rssi[b.name]
            
            if filt_rssi < RSSI_CUTOFF:
                continue
                
            dist = rssi_to_distance(filt_rssi, b.tx)
            points.append((b, dist)); info[b.name] = (filt_rssi, dist)

        raw_pos, label = estimate_position_heatmap(points)
        
        if raw_pos is not None:
            if self.last_pos is not None:
                dx = raw_pos[0] - self.last_pos[0]
                dy = raw_pos[1] - self.last_pos[1]
                dist_moved = math.hypot(dx, dy)

                if dist_moved < 0.5:
                    weight = 0.10  
                elif dist_moved > 5.0:
                    weight = 0.05  
                    label += " (노이즈 방어)"
                else:
                    weight = 0.40  
                    
                pos = (self.last_pos[0] * (1 - weight) + raw_pos[0] * weight,
                       self.last_pos[1] * (1 - weight) + raw_pos[1] * weight)
            else:
                pos = raw_pos
        else:
            pos = self.last_pos
            
        if pos is not None:
            pos = clamp_building(pos)
            
        self.last_pos = pos
        if pos is not None:
            self.trail.append(pos)
            if len(self.trail) > 40: self.trail.pop(0)
            
        return pos, label, len(points), info, err

    def _build_figure(self, headless):
        self.fig, self.ax = plt.subplots(figsize=(13, 8.6))
        self.fig.subplots_adjust(left=0.06, right=0.68, top=0.95, bottom=0.06)
        ax = self.ax; ax.set_xlim(-3, 39.5); ax.set_ylim(-3, 31)
        ax.set_aspect("equal"); ax.grid(True, ls="--", alpha=0.25)
        ax.set_title("용산역 실측 최종본")
        
        ax.add_patch(patches.Polygon([BUILDING["BL"], BUILDING["BR"], BUILDING["TR"], BUILDING["TL"]],
                                     closed=True, fill=False, edgecolor="#1a2f4a", lw=2))
        p = BUILDING["pillar"]
        ax.add_patch(patches.Rectangle((p["x0"], 0), p["x1"] - p["x0"], p["depth"],
                                       facecolor="#c7d0dd", edgecolor="#64748b", lw=1))
        ex0, ex1 = BUILDING["exit"]
        ax.plot([ex0, ex1], [28.2, 28.2], color="tab:green", lw=5, solid_capstyle="round")

        self.range_artists: Dict[str, patches.Circle] = {}
        self.marker_artists: Dict[str, plt.Line2D] = {}
        self.label_artists: Dict[str, plt.Text] = {}
        self.sel_ring, = ax.plot([], [], "o", ms=22, mfc="none", mec="#eab308", mew=2, zorder=8)
        self.pos_marker, = ax.plot([], [], "o", ms=15, color="tab:red", mec="black", zorder=9, label="내 위치")
        self.trail_line, = ax.plot([], [], "-", color="tab:red", alpha=0.35, lw=1.5, zorder=6)
        self.drag_text = ax.text(0, 0, "", fontsize=9, color="#b45309", fontweight="bold",
                                 ha="left", va="bottom", zorder=10,
                                 bbox=dict(boxstyle="round", fc="#fffbe6", ec="#eab308", alpha=0.9))
        self.drag_text.set_visible(False)
        
        self.status = self.fig.text(0.69, 0.94, "", fontsize=9, va="top")
        self.flash = self.fig.text(0.69, 0.10, "", fontsize=10, color="tab:green", va="top", fontweight="bold")
        self._rebuild_static()

        def mkbtn(col, row, label, cb):
            x = 0.69 if col == 0 else 0.83; y = 0.255 - row * 0.062
            axb = self.fig.add_axes([x, y, 0.13, 0.05]); b = Button(axb, label); b.on_clicked(cb); return b
        self.b_add   = mkbtn(0, 0, "추가", self.add_beacon)
        self.b_del   = mkbtn(1, 0, "삭제", self._toggle_delete)
        self.b_cal   = mkbtn(0, 1, "1m보정", lambda e: self.calibrate_selected())
        self.b_save  = mkbtn(1, 1, "저장", self.save)
        self.b_reset = mkbtn(0, 2, "초기화", self.reset)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _rebuild_static(self):
        for d in (self.range_artists, self.marker_artists, self.label_artists):
            for a in d.values():
                try: a.remove()
                except Exception: pass
            d.clear()
        for b in self.active():
            col = "tab:purple" if b.kind == "center" else "tab:blue"
            self.range_artists[b.name] = self.ax.add_patch(
                patches.Circle(tuple(b.xy), 10.0, fill=False, ls=":", alpha=0.15, ec=col, zorder=1))
            m, = self.ax.plot([b.xy[0]], [b.xy[1]], "^", ms=13, color=col, mec="black", zorder=7)
            self.marker_artists[b.name] = m
            self.label_artists[b.name] = self.ax.text(b.xy[0] + 0.6, b.xy[1] + 0.6, b.name,
                                                      fontsize=8.5, fontweight="bold", zorder=7)
        if hasattr(self, "b_add"): self.b_add.label.set_text(f"추가 {len(self.active())}/{MAX_ACTIVE}")

    def _toggle_delete(self, *_):
        self.delete_mode = not self.delete_mode
        self.b_del.label.set_text("삭제 ●" if self.delete_mode else "삭제")

    def _flash(self, msg): self.flash.set_text(msg)

    def _pick(self, x, y):
        best, bd = None, DRAG_PICK_RADIUS
        for b in self.active():
            d = math.hypot(x - b.xy[0], y - b.xy[1])
            if d < bd: best, bd = b, d
        return best

    def _on_press(self, e):
        if e.inaxes != self.ax or e.xdata is None: return
        b = self._pick(e.xdata, e.ydata)
        if b is None: return
        self.selected = b.name
        if self.delete_mode: self.remove_selected(); return
        self.drag_name = b.name

    def _on_motion(self, e):
        if self.drag_name is None or e.inaxes != self.ax or e.xdata is None: return
        x = min(max(round(e.xdata / SNAP) * SNAP, 0), BUILDING["W_bottom"])
        y = min(max(round(e.ydata / SNAP) * SNAP, 0), BUILDING["H"])
        for b in self.beacons:
            if b.name == self.drag_name:
                b.xy = [x, y]; self.marker_artists[b.name].set_data([x], [y])
                self.label_artists[b.name].set_position((x + 0.6, y + 0.6))
                self.range_artists[b.name].center = (x, y)
                self.drag_text.set_position((x + 0.8, y + 1.0))
                self.drag_text.set_text(f"{b.name} ({x:.3f}, {y:.3f})") 
                self.drag_text.set_visible(True); break

    def _on_release(self, e):
        if self.drag_name is not None: self.drag_name = None; self.drag_text.set_visible(False)

    def _on_key(self, e):
        k = (e.key or "").lower()
        if k == "a": self.add_beacon()
        elif k == "d": self.remove_selected()
        elif k == "c": self.calibrate_selected()
        elif k == "s": self.save()
        elif k == "r": self.reset()

    def update(self, _frame=None):
        pos, label, nseen, info, err = self._compute()
        if pos is not None:
            self.pos_marker.set_data([pos[0]], [pos[1]])
            if self.trail: tx, ty = zip(*self.trail); self.trail_line.set_data(tx, ty)
        else:
            self.pos_marker.set_data([], []); self.trail_line.set_data([], [])
            
        sel = next((b for b in self.active() if b.name == self.selected), None)
        if sel: self.sel_ring.set_data([sel.xy[0]], [sel.xy[1]])
        else:   self.sel_ring.set_data([], [])
        
        posent = f"({pos[0]:.3f}, {pos[1]:.3f}) m" if pos else "—"
        
        lines = [
            f"■ 내 실시간 위치: {posent}", 
            f"■ 위치 계산 방식: {label}",
            f"■ 수신된 비콘 수: {nseen}/{len(self.active())}", 
            "", 
            "비콘명   [고정 설치 좌표]   측정거리   신호(RSSI)"
        ]
        
        for b in self.active():
            if b.name in info: 
                rssi, dist = info[b.name]
                lines.append(f"{b.name:<7} ({b.xy[0]:5.1f}, {b.xy[1]:5.1f})   {dist:5.2f}m   {rssi:5.0f}dBm")
            else: 
                lines.append(f"{b.name:<7} ({b.xy[0]:5.1f}, {b.xy[1]:5.1f})   신호 무시됨 (노이즈/멀어짐)")
                
        self.status.set_text("\n".join(lines))
        return ()

    def run(self):
        self.receiver.start()
        self.anim = FuncAnimation(self.fig, self.update, interval=100, blit=False, cache_frame_data=False)
        plt.show()

if __name__ == "__main__":
    TrackerApp().run()