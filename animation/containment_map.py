"""
Containment Error Animation (light mode, map only)
====================================================
Shows 4 follower agents converging into the convex hull spanned by 4 fixed
leader agents, using the fixed-time containment controller (Theorem 2, Eq. 23)
from:

    Su et al. (2024) – "A Fixed-Time Formation-Containment Control Scheme
    for Multi-Agent Systems With Motion Planning: Applications to Quadcopter
    UAVs", IEEE Transactions on Vehicular Technology, vol. 73, no. 7.

─── Leaders (FIXED – already in diamond formation) ────────────────────────────
    L1 = [ 0,  2]   L2 = [ 2,  0]   L3 = [ 0, -2]   L4 = [-2,  0]

─── Adjacency matrix (followers communicate with leaders AND other followers) ──
  Columns: [L1=0, L2=1, L3=2, L4=3, F1=4, F2=5, F3=6, F4=7]
  A_f[k, j] > 0  ⟹  agent j is an in-neighbour of follower k

    F1 ← L1, F2, F4     γ_1 = 3
    F2 ← L2, F1, F3     γ_2 = 3
    F3 ← L3, F2, F4     γ_3 = 3
    F4 ← L4, F1, F3     γ_4 = 3

  Follower graph: F1–F2–F3–F4–F1 (bidirectional ring).
  Each follower also connects directly to one leader → Assumption 1 satisfied
  (directed path from virtual leader → leader → follower for all followers).

  Full Laplacian L_ff for follower subsystem (diagonal = γ_k = 3, off-diag = −A_ff):
    eigenvalues = [1, 3, 3, 5]  → all positive → globally stable ✓

─── Coupled equilibrium (where ξ_k = 0 simultaneously for all k) ──────────────
  Solving  3·p_Fk = p_leader + p_Fj + p_Fm  for all k simultaneously:

    F1* = [ 0,    2/3]
    F2* = [ 2/3,  0  ]
    F3* = [ 0,   -2/3]
    F4* = [-2/3,  0  ]

  These form a smaller inner diamond (radius 2/3 vs 2 for leaders).
  All points satisfy |x|+|y| = 2/3 ≤ 2  →  inside convex hull ✓

─── Containment error (Eq. 4) ──────────────────────────────────────────────────
  ξ_k = Σ_{j∈N_k} a_{kj} (p_k − p_j)

─── Control law (Theorem 2, Eq. 23) ────────────────────────────────────────────
  u_k = −1/γ_k [(α + β·μ̇/μ)·ξ_k + (α + β·μ̇/μ)·ζ_k]

  ζ_k = Σ_{j∈N_k} a_{kj} (v_k − v_j)   [velocity error, Eq. 6]
  μ̇/μ = h/(T−t)  for t < T
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# ── Simulation parameters ────────────────────────────────────────────────────
np.random.seed(42)

N_L       = 4        # number of leaders  (fixed)
N_F       = 4        # number of followers (moving)
DT        = 0.05
T_SIM     = 20.0
T_CONV    = 12.0
ALPHA     = 4.0      # large enough for zeta_min ≈ 1.0 (critically damped)
BETA      = 0.3
H_EXP     = 5.0
GAIN_CAP  = 8.0      # cap prevents Euler instability near T (λ_max=5, DT=0.05)

# ── Leader positions (fixed) ──────────────────────────────────────────────────
P_LEADERS = np.array([
    [ 0.0,  2.0],   # L1
    [ 2.0,  0.0],   # L2
    [ 0.0, -2.0],   # L3
    [-2.0,  0.0],   # L4
])
V_LEADERS = np.zeros((N_L, 2))

# ── Follower initial positions ────────────────────────────────────────────────
P_FOL_0 = (np.random.rand(N_F, 2) - 0.5) * 9.0

# ── Adjacency matrix ─────────────────────────────────────────────────────────
# Columns: [L1=0, L2=1, L3=2, L4=3, F1=4, F2=5, F3=6, F4=7]
A_f = np.zeros((N_F, N_L + N_F))

# Leader connections (each follower ← one dedicated leader)
A_f[0, 0] = 1.0   # F1 ← L1
A_f[1, 1] = 1.0   # F2 ← L2
A_f[2, 2] = 1.0   # F3 ← L3
A_f[3, 3] = 1.0   # F4 ← L4

# Follower-to-follower connections (bidirectional ring F1–F2–F3–F4–F1)
A_f[0, 5] = 1.0   # F1 ← F2
A_f[0, 7] = 1.0   # F1 ← F4
A_f[1, 4] = 1.0   # F2 ← F1
A_f[1, 6] = 1.0   # F2 ← F3
A_f[2, 5] = 1.0   # F3 ← F2
A_f[2, 7] = 1.0   # F3 ← F4
A_f[3, 4] = 1.0   # F4 ← F1
A_f[3, 6] = 1.0   # F4 ← F3

GAMMA_F = A_f.sum(axis=1)   # γ_k = 3 for all followers

# Coupled equilibrium positions (solved analytically)
EQ_TARGETS = np.array([
    [ 0.0,    2/3],   # F1*
    [ 2/3,    0.0],   # F2*
    [ 0.0,   -2/3],   # F3*
    [-2/3,    0.0],   # F4*
])

# ── Time-varying function ─────────────────────────────────────────────────────
def mu_dot_ratio(t: float) -> float:
    return H_EXP / (T_CONV - t) if t < T_CONV - 1e-6 else 0.0


# ── Simulation ────────────────────────────────────────────────────────────────
class ContainmentSim:
    def __init__(self):
        self.p_f = P_FOL_0.copy()
        self.v_f = np.zeros((N_F, 2))
        self.t   = 0.0

        n = int(T_SIM / DT) + 1
        self.t_hist  = np.zeros(n)
        self.pf_hist = np.zeros((n, N_F, 2))
        self.xi_hist = np.zeros((n, N_F))
        self._record(0)

    def _all_p(self):
        return np.vstack([P_LEADERS, self.p_f])

    def _all_v(self):
        return np.vstack([V_LEADERS, self.v_f])

    def _xi_vec(self, k: int) -> np.ndarray:
        """ξ_k = Σ_{j∈N_k} a_{kj} (p_k − p_j)  [Eq. 4]"""
        p_all = self._all_p()
        xi = np.zeros(2)
        for j in range(N_L + N_F):
            if A_f[k, j] > 0:
                xi += A_f[k, j] * (self.p_f[k] - p_all[j])
        return xi

    def _zeta_vec(self, k: int) -> np.ndarray:
        """ζ_k = Σ_{j∈N_k} a_{kj} (v_k − v_j)  [Eq. 6]"""
        v_all = self._all_v()
        zeta = np.zeros(2)
        for j in range(N_L + N_F):
            if A_f[k, j] > 0:
                zeta += A_f[k, j] * (self.v_f[k] - v_all[j])
        return zeta

    def _control(self, k: int) -> np.ndarray:
        """u_k = −1/γ_k [(α + β·μ̇/μ)·ξ_k + (α + β·μ̇/μ)·ζ_k]
        Gain is capped at GAIN_CAP to keep Euler integration stable near t=T
        (the λ_max=5 mode requires gain·λ·DT < 2 for stability)."""
        gain = min(ALPHA + BETA * mu_dot_ratio(self.t), GAIN_CAP)
        return -(1.0 / GAMMA_F[k]) * (gain * self._xi_vec(k) + gain * self._zeta_vec(k))

    def _record(self, idx: int):
        self.t_hist[idx]  = self.t
        self.pf_hist[idx] = self.p_f.copy()
        self.xi_hist[idx] = np.array([np.linalg.norm(self._xi_vec(k)) for k in range(N_F)])

    def step(self, idx: int):
        accel    = np.array([self._control(k) for k in range(N_F)])
        self.v_f += accel * DT
        self.p_f += self.v_f * DT
        self.t   += DT
        self._record(idx)

    def run_all(self) -> int:
        n = int(T_SIM / DT)
        for k in range(1, n + 1):
            self.step(k)
        return n + 1


# ── Pre-compute ───────────────────────────────────────────────────────────────
print("Running containment simulation…")
sim      = ContainmentSim()
n_frames = sim.run_all()
print(f"  {n_frames} frames ready")
print(f"  Follower initial positions:\n{P_FOL_0.round(2)}")
print(f"  Initial containment errors : {sim.xi_hist[0].round(3)}")
print(f"  At T_CONV (t={T_CONV}s)    : {sim.xi_hist[int(T_CONV/DT)].round(4)}")
print(f"  Final errors  (t={T_SIM}s) : {sim.xi_hist[-1].round(4)}")

# ── Colour palette (light mode) ───────────────────────────────────────────────
L_COLORS = ['#1565c0', '#e65100', '#2e7d32', '#ad1457']   # leaders
F_COLORS = ['#6a1b9a', '#00695c', '#f57f17', '#37474f']   # followers
L_LABELS = ['L1', 'L2', 'L3', 'L4']
F_LABELS = ['F1', 'F2', 'F3', 'F4']

VL_COLOR   = '#00695c'
HULL_COLOR = '#bbdefb'
HULL_EDGE  = '#90caf9'
GRID_COL   = '#e0e0e0'
SPINE_COL  = '#bdbdbd'
TEXT_COL   = '#333333'

SKIP = 2

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
ax.set_facecolor('white')
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True, color=GRID_COL, lw=0.8, zorder=0)
ax.set_xlabel('X (m)', fontsize=12, color=TEXT_COL)
ax.set_ylabel('Y (m)', fontsize=12, color=TEXT_COL)
ax.set_title(
    'Fixed-Time Containment Control\n'
    r'Followers converge into convex hull  '
    r'(topology: $F_k \leftarrow$ leader $+$ two adjacent followers)',
    fontsize=11, color='#222222', pad=10
)
ax.tick_params(colors='#444444')
for sp in ax.spines.values():
    sp.set_edgecolor(SPINE_COL)

# ── Static elements ───────────────────────────────────────────────────────────

# Convex hull (diamond polygon)
hull_patch = Polygon(P_LEADERS[[0, 1, 2, 3]], closed=True,
                     facecolor=HULL_COLOR, edgecolor=HULL_EDGE,
                     lw=1.5, alpha=0.50, zorder=1)
ax.add_patch(hull_patch)

# Inner diamond (equilibrium targets outline)
inner_pts = EQ_TARGETS[[0, 1, 2, 3]]
inner_patch = Polygon(inner_pts, closed=True,
                      facecolor='none', edgecolor='#9e9e9e',
                      lw=1.0, linestyle='--', alpha=0.5, zorder=2)
ax.add_patch(inner_patch)

# Equilibrium target markers
for k in range(N_F):
    ax.plot(*EQ_TARGETS[k], 'D', ms=6, color=F_COLORS[k],
            alpha=0.45, zorder=3, markeredgewidth=0)

# Virtual leader
ax.plot(0, 0, '*', ms=20, color=VL_COLOR, zorder=10,
        markeredgecolor='white', markeredgewidth=0.5)
ax.text(0.18, 0.18, 'V', color=VL_COLOR, fontsize=9,
        fontweight='bold', ha='center', va='bottom', zorder=11)

# Fixed leader markers
for i in range(N_L):
    ax.plot(*P_LEADERS[i], '^', ms=14, color=L_COLORS[i],
            markeredgecolor='white', markeredgewidth=0.8, zorder=8)
    ax.text(P_LEADERS[i, 0], P_LEADERS[i, 1] + 0.30, L_LABELS[i],
            color=L_COLORS[i], fontsize=9, fontweight='bold',
            ha='center', va='bottom', zorder=9)

# Static topology: leader → follower equilibrium target (very faint arrows)
# L1→F1*, L2→F2*, L3→F3*, L4→F4*
for i in range(N_L):
    ax.annotate('', xy=EQ_TARGETS[i], xytext=P_LEADERS[i],
                arrowprops=dict(arrowstyle='->', color=L_COLORS[i],
                                lw=0.8, alpha=0.18, linestyle='dashed'),
                zorder=3)

# ── Animated elements ─────────────────────────────────────────────────────────

# Follower trajectory traces
traj_lines = [
    ax.plot([], [], '-', color=F_COLORS[k], alpha=0.35, lw=1.3, zorder=4)[0]
    for k in range(N_F)
]

# Live leader → follower arrows (tail fixed at leader, head at follower)
lf_arrows = [
    ax.annotate('', xy=(0.01, 0.01), xytext=P_LEADERS[k],
                arrowprops=dict(arrowstyle='->', color=L_COLORS[k],
                                lw=1.1, alpha=0.55),
                zorder=5)
    for k in range(N_F)   # L1→F1, L2→F2, L3→F3, L4→F4
]

# Live follower ↔ follower communication lines (ring: F1-F2-F3-F4-F1)
# Drawn as thin dashed colored lines; both endpoints move each frame
FF_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]   # (Fi, Fj) index pairs
ff_lines = [
    ax.plot([], [], '--', color='#9e9e9e', lw=0.9, alpha=0.65, zorder=5)[0]
    for _ in FF_PAIRS
]

# Follower position markers (circles)
fol_dots = [
    ax.plot([], [], 'o', ms=12, color=F_COLORS[k],
            markeredgecolor='white', markeredgewidth=0.8, zorder=8)[0]
    for k in range(N_F)
]

# Follower labels
fol_lbls = [
    ax.text(0, 0, F_LABELS[k], color=F_COLORS[k], fontsize=9,
            fontweight='bold', ha='center', va='bottom', zorder=9)
    for k in range(N_F)
]

# Time stamp
time_text = ax.text(0.03, 0.97, '', transform=ax.transAxes,
                    color=TEXT_COL, fontsize=11, va='top',
                    fontfamily='monospace', zorder=12)

# Convergence badge
conv_badge = ax.text(0.97, 0.97, '', transform=ax.transAxes,
                     color='#2e7d32', fontsize=10, va='top', ha='right',
                     fontweight='bold', zorder=12)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elems = [
    mpatches.Patch(facecolor=HULL_COLOR, edgecolor=HULL_EDGE,
                   label='Convex hull of leaders'),
    mpatches.Patch(facecolor='none', edgecolor='#9e9e9e', linestyle='--',
                   label='Equilibrium inner diamond'),
    Line2D([0], [0], marker='*', ms=12, color=VL_COLOR,
           lw=0, markeredgecolor='white', label='V – virtual leader'),
] + [
    Line2D([0], [0], marker='^', ms=9, color=L_COLORS[i],
           lw=0, markeredgecolor='white', label=L_LABELS[i])
    for i in range(N_L)
] + [
    Line2D([0], [0], marker='o', ms=9, color=F_COLORS[k],
           lw=0, markeredgecolor='white', label=F_LABELS[k])
    for k in range(N_F)
] + [
    Line2D([0], [0], linestyle='--', color='#9e9e9e', lw=1.2,
           label='F–F communication link'),
]
ax.legend(handles=legend_elems, loc='lower right',
          facecolor='white', edgecolor=SPINE_COL, fontsize=8,
          framealpha=0.92, ncol=2)

plt.tight_layout()

# ── Animation update ──────────────────────────────────────────────────────────
anim_frames = list(range(0, n_frames, SKIP))


def update(frame_idx: int):
    k     = anim_frames[frame_idx]
    t_now = sim.t_hist[k]
    p_now = sim.pf_hist[k]   # (N_F, 2)

    # Trajectory traces
    for fi in range(N_F):
        traj_lines[fi].set_data(sim.pf_hist[:k+1, fi, 0],
                                sim.pf_hist[:k+1, fi, 1])

    # Leader → follower arrows (head moves with follower)
    for fi in range(N_F):
        lf_arrows[fi].xy = (p_now[fi, 0], p_now[fi, 1])

    # Follower ↔ follower ring lines (both endpoints move)
    for idx, (fi, fj) in enumerate(FF_PAIRS):
        xs = [p_now[fi, 0], p_now[fj, 0]]
        ys = [p_now[fi, 1], p_now[fj, 1]]
        ff_lines[idx].set_data(xs, ys)

    # Follower markers + labels
    for fi in range(N_F):
        fol_dots[fi].set_data([p_now[fi, 0]], [p_now[fi, 1]])
        fol_lbls[fi].set_position((p_now[fi, 0], p_now[fi, 1] + 0.28))

    time_text.set_text(f't = {t_now:5.2f} s')
    conv_badge.set_text('✓ Contained' if t_now >= T_CONV else '')

    return traj_lines + ff_lines + fol_dots + fol_lbls + [time_text, conv_badge]


# ── Save ──────────────────────────────────────────────────────────────────────
anim = FuncAnimation(
    fig, update,
    frames=len(anim_frames),
    interval=50,
    blit=False,
    repeat=True,
)

output_path = 'animation/containment_map.gif'
print(f"Saving GIF → {output_path}  ({len(anim_frames)} frames) …")
anim.save(output_path, writer='pillow', fps=20, dpi=120)
print("Done →", output_path)
