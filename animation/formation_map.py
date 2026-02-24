"""
Formation Map Animation (light mode, map only)
===============================================
Same simulation as formation_error.py but rendered as a single spatial
panel with a white presentation-friendly theme.

Control law, topology, and formation offsets are identical:
  – 4 leaders, static virtual leader at origin
  – Star topology: a_{i,V} = 1 for each leader
  – Diamond formation offsets: h = {[0,2], [2,0], [0,-2], [-2,0]}
  – Fixed-time controller (Theorem 2, Eq. 23)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# ── Simulation parameters (identical to formation_error.py) ─────────────────
np.random.seed(7)

N      = 4
DT     = 0.05
T_SIM  = 30.0
T_CONV = 20.0
ALPHA  = 0.05
BETA   = 0.4
H_EXP  = 5.0

H = np.array([
    [ 0.0,  2.0],   # L1 – top
    [ 2.0,  0.0],   # L2 – right
    [ 0.0, -2.0],   # L3 – bottom
    [-2.0,  0.0],   # L4 – left
])

A = np.zeros((N, N + 1))
for i in range(N):
    A[i, N] = 1.0
GAMMA = A.sum(axis=1)


def mu_dot_ratio(t: float) -> float:
    return H_EXP / (T_CONV - t) if t < T_CONV - 1e-6 else 0.0


class FormationSim:
    def __init__(self):
        self.p    = (np.random.rand(N, 2) - 0.5) * 8.0
        self.v    = np.zeros((N, 2))
        self.p_vl = np.zeros(2)
        self.v_vl = np.zeros(2)
        self.t    = 0.0

        n = int(T_SIM / DT) + 1
        self.t_hist  = np.zeros(n)
        self.p_hist  = np.zeros((n, N, 2))
        self.xi_hist = np.zeros((n, N))
        self._record(0)

    def _xi_vec(self, i):
        return A[i, N] * (self.p[i] - self.p_vl - H[i])

    def _zeta_vec(self, i):
        return A[i, N] * (self.v[i] - self.v_vl)

    def _xi_norms(self):
        return np.array([np.linalg.norm(self._xi_vec(i)) for i in range(N)])

    def _control(self, i):
        gain = ALPHA + BETA * mu_dot_ratio(self.t)
        return -(1.0 / GAMMA[i]) * (gain * self._xi_vec(i) + gain * self._zeta_vec(i))

    def _record(self, idx):
        self.t_hist[idx]  = self.t
        self.p_hist[idx]  = self.p.copy()
        self.xi_hist[idx] = self._xi_norms()

    def step(self, idx):
        accel   = np.array([self._control(i) for i in range(N)])
        self.v += accel * DT
        self.p += self.v * DT
        self.t += DT
        self._record(idx)

    def run_all(self):
        n = int(T_SIM / DT)
        for k in range(1, n + 1):
            self.step(k)
        return n + 1


# ── Pre-compute ──────────────────────────────────────────────────────────────
print("Running formation simulation…")
sim      = FormationSim()
n_frames = sim.run_all()
print(f"  {n_frames} frames ready")

desired = H.copy()

# ── Light-mode colour palette ────────────────────────────────────────────────
COLORS = ['#1565c0', '#e65100', '#2e7d32', '#ad1457']   # deep blue/orange/green/pink
LABELS = ['L1', 'L2', 'L3', 'L4']

VL_COLOR   = '#00695c'   # teal star for virtual leader
TRAJ_ALPHA = 0.40
GRID_COLOR = '#e0e0e0'
SPINE_COL  = '#bdbdbd'

SKIP = 2   # sub-sample: every 2nd frame → 20 fps

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
ax.set_facecolor('white')
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True, color=GRID_COLOR, lw=0.8, zorder=0)
ax.set_xlabel('X (m)', fontsize=12, color='#333333')
ax.set_ylabel('Y (m)', fontsize=12, color='#333333')
ax.set_title(
    'Fixed-Time Formation Tracking\n'
    r'Leaders $\rightarrow$ Diamond Formation around Virtual Leader $V$',
    fontsize=12, color='#222222', pad=10
)
ax.tick_params(colors='#444444')
for sp in ax.spines.values():
    sp.set_edgecolor(SPINE_COL)

# ── Static elements ───────────────────────────────────────────────────────────

# Target diamond outline
diamond_pts = np.vstack([desired, desired[0]])
ax.plot(diamond_pts[:, 0], diamond_pts[:, 1],
        '--', color='#9e9e9e', lw=1.2, alpha=0.6, zorder=1)

# Desired position markers (×)
for i in range(N):
    ax.plot(*desired[i], 'x', ms=12, mew=2.2, color=COLORS[i], alpha=0.5, zorder=2)

# Virtual leader (star)
ax.plot(0, 0, '*', ms=22, color=VL_COLOR, zorder=10,
        markeredgecolor='white', markeredgewidth=0.5)

# Static topology arrows: V → desired positions (faint)
for i in range(N):
    ax.annotate('', xy=desired[i], xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=COLORS[i],
                                lw=0.9, alpha=0.18),
                zorder=3)

# ── Animated elements ─────────────────────────────────────────────────────────

# Trajectory traces
traj_lines = [
    ax.plot([], [], '-', color=COLORS[i], alpha=TRAJ_ALPHA, lw=1.4, zorder=4)[0]
    for i in range(N)
]

# Live communication edges  V → current leader position
comm_arrows = [
    ax.annotate('', xy=(0.01, 0.01), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=COLORS[i],
                                lw=1.0, alpha=0.55),
                zorder=5)
    for i in range(N)
]

# Current position markers (triangles)
leader_dots = [
    ax.plot([], [], '^', ms=13, color=COLORS[i],
            markeredgecolor='white', markeredgewidth=0.8,
            zorder=8)[0]
    for i in range(N)
]

# Agent labels
leader_lbls = [
    ax.text(0, 0, LABELS[i], color=COLORS[i], fontsize=9,
            fontweight='bold', ha='center', va='bottom', zorder=9)
    for i in range(N)
]

# Virtual leader label
ax.text(0.18, 0.18, 'V', color=VL_COLOR, fontsize=10,
        fontweight='bold', ha='center', va='bottom', zorder=11)

# Time stamp (top-left)
time_text = ax.text(0.03, 0.97, '', transform=ax.transAxes,
                    color='#333333', fontsize=11, va='top',
                    fontfamily='monospace', zorder=12)

# Convergence badge (top-right)
conv_badge = ax.text(0.97, 0.97, '', transform=ax.transAxes,
                     color='#2e7d32', fontsize=10, va='top', ha='right',
                     fontweight='bold', zorder=12)

# T label near convergence time indicator (added as a horizontal guide line on map)
# Show a subtle "t = T" annotation once converged

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elems = [
    Line2D([0], [0], marker='*', ms=13, color=VL_COLOR,
           lw=0, markeredgecolor='white', label='V – virtual leader'),
] + [
    Line2D([0], [0], marker='^', ms=9, color=COLORS[i],
           lw=0, markeredgecolor='white', label=LABELS[i])
    for i in range(N)
] + [
    Line2D([0], [0], marker='x', ms=8, color='#757575',
           lw=0, markeredgewidth=2, label='Target positions'),
]
ax.legend(handles=legend_elems, loc='lower right',
          facecolor='white', edgecolor=SPINE_COL, fontsize=9,
          framealpha=0.9)

plt.tight_layout()

# ── Animation update ──────────────────────────────────────────────────────────
anim_frames = list(range(0, n_frames, SKIP))


def update(frame_idx: int):
    k     = anim_frames[frame_idx]
    t_now = sim.t_hist[k]
    p_now = sim.p_hist[k]

    # Trajectory traces
    for i in range(N):
        traj_lines[i].set_data(sim.p_hist[:k+1, i, 0],
                               sim.p_hist[:k+1, i, 1])

    # Leader markers + labels
    for i in range(N):
        leader_dots[i].set_data([p_now[i, 0]], [p_now[i, 1]])
        leader_lbls[i].set_position((p_now[i, 0], p_now[i, 1] + 0.28))

    # Live communication arrows  V → Li
    for i in range(N):
        comm_arrows[i].xy = (p_now[i, 0], p_now[i, 1])

    # Time stamp
    time_text.set_text(f't = {t_now:5.2f} s')

    # Convergence badge
    conv_badge.set_text('✓ Converged' if t_now >= T_CONV else '')

    return traj_lines + leader_dots + leader_lbls + [time_text, conv_badge]


# ── Save ──────────────────────────────────────────────────────────────────────
anim = FuncAnimation(
    fig, update,
    frames=len(anim_frames),
    interval=50,
    blit=False,
    repeat=True,
)

output_path = 'animation/formation_map.gif'
print(f"Saving GIF → {output_path}  ({len(anim_frames)} frames) …")
anim.save(output_path, writer='pillow', fps=20, dpi=120)
print("Done →", output_path)
