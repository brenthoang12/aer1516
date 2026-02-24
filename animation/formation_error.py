"""
Formation Error Animation
=========================
Animates 4 leader agents converging from random initial positions to a
diamond formation centred on a static virtual leader, using the fixed-time
formation-tracking control law (Theorem 2, Eq. 23) from:

    Su et al. (2024) – "A Fixed-Time Formation-Containment Control Scheme
    for Multi-Agent Systems With Motion Planning: Applications to Quadcopter
    UAVs", IEEE Transactions on Vehicular Technology, vol. 73, no. 7.

─── Agents ────────────────────────────────────────────────────────────────────
  V  – virtual leader  (index 4, static at origin)
  L1 – leader 1        (index 0)
  L2 – leader 2        (index 1)
  L3 – leader 3        (index 2)
  L4 – leader 4        (index 3)

─── Desired formation offsets h_i (position of Li relative to V) ──────────────
  h_1 = [ 0,  2]  → L1 above V
  h_2 = [ 2,  0]  → L2 right of V
  h_3 = [ 0, -2]  → L3 below V
  h_4 = [-2,  0]  → L4 left of V
  (diamond shape, 2 m radius)

─── Adjacency matrix A (star topology) ────────────────────────────────────────
  Every leader receives information directly from the virtual leader.
  No inter-leader edges.

    A[i, V] = 1  for i ∈ {0,1,2,3}
    A[i, j] = 0  otherwise

  This satisfies Assumption 1 (directed spanning tree, V as root node).

─── Control law (Theorem 2, Eq. 23) ───────────────────────────────────────────
  u_i = -1/γ_i [(α + β·μ̇/μ)·ξ_i + (α + β·μ̇/μ)·ζ_i]

  where
    γ_i   = Σ_j a_ij = 1
    ξ_i   = a_{i,V}·(p_i − p_V − h_i)   [formation error,   Eq. 3]
    ζ_i   = a_{i,V}·(v_i − v_V)           [velocity error,    Eq. 5]
    μ(t)  = (T/(T−t))^h  for t < T        [time-varying fn,  Eq. 20]
    μ̇/μ  = h/(T−t)       for t < T

  Acceleration feedback Σ a_ij·v̇_j = 0  (virtual leader is static).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# ── Simulation parameters ────────────────────────────────────────────────────
np.random.seed(7)        # reproducible random initial positions

N      = 4               # number of leader agents
DT     = 0.05            # integration time step  (s)
T_SIM  = 30.0            # total simulation time   (s)
T_CONV = 20.0            # fixed convergence time T (s)
ALPHA  = 0.05            # α  (control gain)
BETA   = 0.4             # β  (control gain)
H_EXP  = 5.0             # h  (exponent in μ(t))

# ── Formation offsets h_i ────────────────────────────────────────────────────
# Desired position of leader i  =  p_V + H[i]
# (p_V = [0,0] so absolute desired positions equal H[i])
H = np.array([
    [ 0.0,  2.0],   # L1 – top
    [ 2.0,  0.0],   # L2 – right
    [ 0.0, -2.0],   # L3 – bottom
    [-2.0,  0.0],   # L4 – left
])

# ── Adjacency matrix (star topology) ────────────────────────────────────────
# Columns: [L1, L2, L3, L4, V]  (virtual leader has column index N = 4)
# A[i, j] > 0  ⟹  agent j is an in-neighbour of i  (j sends info to i)
A = np.zeros((N, N + 1))
for i in range(N):
    A[i, N] = 1.0           # leader i ← virtual leader

GAMMA = A.sum(axis=1)       # γ_i = Σ_j a_ij  (= 1 for all i in star topology)


# ── Time-varying functions (Eq. 20) ─────────────────────────────────────────
def mu(t: float) -> float:
    """μ(t) = (T/(T−t))^h for t < T, else 0."""
    if t < T_CONV - 1e-6:
        return (T_CONV / (T_CONV - t)) ** H_EXP
    return 0.0


def mu_dot_ratio(t: float) -> float:
    """μ̇(t)/μ(t) = h/(T−t) for t < T, else 0."""
    if t < T_CONV - 1e-6:
        return H_EXP / (T_CONV - t)
    return 0.0


# ── Simulation class ─────────────────────────────────────────────────────────
class FormationSim:
    """
    Double-integrator dynamics for N leader agents converging to formation.
    Virtual leader is static at the origin.
    All data is pre-computed and stored for smooth animation playback.
    """

    def __init__(self):
        # Random initial positions in the range [−4, 4] × [−4, 4] (m)
        self.p = (np.random.rand(N, 2) - 0.5) * 8.0   # (N, 2)
        self.v = np.zeros((N, 2))                       # (N, 2)

        # Virtual leader – static at origin
        self.p_vl = np.zeros(2)
        self.v_vl = np.zeros(2)

        self.t = 0.0

        # Pre-allocated history arrays
        n_steps = int(T_SIM / DT) + 1
        self.t_hist  = np.zeros(n_steps)
        self.p_hist  = np.zeros((n_steps, N, 2))  # positions over time
        self.xi_hist = np.zeros((n_steps, N))      # ||ξ_i(t)||

        self._record(0)

    # ── Error vectors ────────────────────────────────────────────────────────

    def _xi_vec(self, i: int) -> np.ndarray:
        """
        Formation tracking error for leader i  (Eq. 3):
            ξ_i = Σ_{j∈N_i} a_{ij} (p_i − p_j − δ_{ij})

        With star topology (only in-neighbour is V) and δ_{i,V} = h_i − h_V = h_i:
            ξ_i = a_{i,V} · (p_i − p_V − h_i)
        """
        return A[i, N] * (self.p[i] - self.p_vl - H[i])

    def _zeta_vec(self, i: int) -> np.ndarray:
        """
        Velocity error for leader i  (Eq. 5):
            ζ_i = Σ_{j∈N_i} a_{ij} (v_i − v_j)
                = a_{i,V} · (v_i − v_V)
        """
        return A[i, N] * (self.v[i] - self.v_vl)

    def _xi_norms(self) -> np.ndarray:
        return np.array([np.linalg.norm(self._xi_vec(i)) for i in range(N)])

    # ── Control law (Theorem 2, Eq. 23) ─────────────────────────────────────

    def _control(self, i: int) -> np.ndarray:
        """
        Fixed-time formation tracking control:
            u_i = −1/γ_i [(α + β·μ̇/μ)·ξ_i + (α + β·μ̇/μ)·ζ_i]

        (Acceleration feedback = 0 because virtual leader is static.)
        """
        gain = ALPHA + BETA * mu_dot_ratio(self.t)
        xi   = self._xi_vec(i)
        zeta = self._zeta_vec(i)
        return -(1.0 / GAMMA[i]) * (gain * xi + gain * zeta)

    # ── Integration ─────────────────────────────────────────────────────────

    def _record(self, idx: int):
        self.t_hist[idx]  = self.t
        self.p_hist[idx]  = self.p.copy()
        self.xi_hist[idx] = self._xi_norms()

    def step(self, store_idx: int):
        """One Euler step:  v̇ = u,  ṗ = v."""
        accel    = np.array([self._control(i) for i in range(N)])
        self.v  += accel * DT
        self.p  += self.v * DT
        self.t  += DT
        self._record(store_idx)

    def run_all(self) -> int:
        """Pre-compute the full simulation; returns total frame count."""
        n_steps = int(T_SIM / DT)
        for k in range(1, n_steps + 1):
            self.step(k)
        return n_steps + 1


# ── Pre-compute simulation ───────────────────────────────────────────────────
print("Running formation simulation…")
sim      = FormationSim()
n_frames = sim.run_all()
print(f"  {n_frames} frames computed  (T_SIM={T_SIM}s, DT={DT}s)")
print(f"  Initial formation errors: {sim.xi_hist[0]}")
print(f"  Final   formation errors: {sim.xi_hist[-1]}")

desired = H.copy()   # absolute desired positions (= p_V + H, with p_V = [0,0])

# ── Colour / style ───────────────────────────────────────────────────────────
COLORS = ['#4fc3f7', '#ffb74d', '#81c784', '#f06292']   # L1..L4 (pastel)
LABELS = ['L1', 'L2', 'L3', 'L4']

BG_DARK  = '#0d1117'
PANEL_BG = '#161b22'
GRID_COL = '#30363d'
TEXT_COL = '#e6edf3'

# Sub-sample frames for animation speed
SKIP = 2   # every 2nd frame → 20 fps at interval=50 ms

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6.5), facecolor=BG_DARK)
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.38,
                        left=0.07, right=0.97, top=0.91, bottom=0.11)

ax_sp = fig.add_subplot(gs[0])   # spatial view
ax_er = fig.add_subplot(gs[1])   # error plot

fig.suptitle(
    "Fixed-Time Formation Tracking  –  Leaders Converging to Diamond Formation\n"
    r"(Theorem 2, Eq. 23:  $u_i = -\frac{1}{\gamma_i}"
    r"[(\alpha+\beta\dot{\mu}/\mu)\xi_i + (\alpha+\beta\dot{\mu}/\mu)\zeta_i]$)",
    color=TEXT_COL, fontsize=11, y=0.99
)

# ── Spatial axes ─────────────────────────────────────────────────────────────
ax_sp.set_facecolor(PANEL_BG)
ax_sp.set_xlim(-6, 6)
ax_sp.set_ylim(-6, 6)
ax_sp.set_aspect('equal')
ax_sp.grid(True, color=GRID_COL, alpha=0.6, lw=0.5)
ax_sp.set_xlabel('X (m)', color=TEXT_COL, fontsize=11)
ax_sp.set_ylabel('Y (m)', color=TEXT_COL, fontsize=11)
ax_sp.set_title('Agent Positions', color=TEXT_COL, fontsize=12, pad=8)
ax_sp.tick_params(colors=TEXT_COL)
for sp in ax_sp.spines.values():
    sp.set_edgecolor(GRID_COL)

# Desired position markers (static)
for i in range(N):
    ax_sp.plot(*desired[i], 'x', ms=11, mew=2, color=COLORS[i], alpha=0.45, zorder=2)

# Target diamond outline (static dashed)
diamond_pts = np.vstack([desired, desired[0]])
ax_sp.plot(diamond_pts[:, 0], diamond_pts[:, 1],
           '--', color='white', alpha=0.15, lw=1.2, zorder=1)

# Virtual leader (static star)
ax_sp.plot(0, 0, '*', ms=20, color='lime', zorder=10, label='V (virtual leader)')

# Communication edges V → desired positions (static, shows topology)
for i in range(N):
    ax_sp.annotate('', xy=desired[i], xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=COLORS[i],
                                   lw=1.0, alpha=0.2),
                   zorder=3)

# Animated: trajectory paths (one per leader)
traj_lines = [
    ax_sp.plot([], [], '-', color=COLORS[i], alpha=0.45, lw=1.2, zorder=4)[0]
    for i in range(N)
]

# Animated: current position markers
leader_dots = [
    ax_sp.plot([], [], '^', ms=12, color=COLORS[i],
               markeredgecolor='white', markeredgewidth=0.7,
               zorder=8, label=LABELS[i])[0]
    for i in range(N)
]

# Animated: text labels next to each leader
leader_lbls = [
    ax_sp.text(0, 0, LABELS[i], color=COLORS[i], fontsize=9,
               fontweight='bold', ha='center', va='bottom', zorder=9)
    for i in range(N)
]

# Animated: live communication edges V → each leader's current position
comm_arrows = [
    ax_sp.annotate('', xy=(0, 0), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=COLORS[i],
                                   lw=0.8, alpha=0.5),
                   zorder=5)
    for i in range(N)
]

# Time label
time_text = ax_sp.text(0.03, 0.97, '', transform=ax_sp.transAxes,
                        color=TEXT_COL, fontsize=11, va='top',
                        fontfamily='monospace', zorder=11)

# Convergence indicator dot (appears at t ≥ T_CONV)
conv_badge = ax_sp.text(0.97, 0.97, '', transform=ax_sp.transAxes,
                         color='lime', fontsize=10, va='top', ha='right',
                         fontweight='bold', zorder=11)

# Spatial legend
sp_legend_elems = [
    Line2D([0], [0], marker='*', ms=13, color='lime',
           lw=0, label='V – virtual leader'),
] + [
    Line2D([0], [0], marker='^', ms=9, color=COLORS[i],
           lw=0, markeredgecolor='white', label=LABELS[i])
    for i in range(N)
] + [
    Line2D([0], [0], marker='x', ms=8, color='grey',
           lw=0, markeredgewidth=2, label='Desired positions'),
]
ax_sp.legend(handles=sp_legend_elems, loc='lower right',
             facecolor='#21262d', edgecolor=GRID_COL, labelcolor=TEXT_COL,
             fontsize=8, framealpha=0.85)

# ── Error axes ────────────────────────────────────────────────────────────────
ax_er.set_facecolor(PANEL_BG)
y_max = sim.xi_hist[0].max() * 1.15 + 0.3
ax_er.set_xlim(0, T_SIM)
ax_er.set_ylim(0, y_max)
ax_er.set_xlabel('Time  (s)', color=TEXT_COL, fontsize=11)
ax_er.set_ylabel(r'$\|\xi_i(t)\|$  (m)', color=TEXT_COL, fontsize=12)
ax_er.set_title('Formation Error per Leader', color=TEXT_COL, fontsize=12, pad=8)
ax_er.grid(True, color=GRID_COL, alpha=0.6, lw=0.5)
ax_er.tick_params(colors=TEXT_COL)
for sp in ax_er.spines.values():
    sp.set_edgecolor(GRID_COL)

# Fixed-time convergence line
ax_er.axvline(x=T_CONV, color='#ff6b6b', linestyle='--', lw=1.5, alpha=0.8,
              zorder=5)
ax_er.text(T_CONV + 0.3, y_max * 0.95,
           f'T = {T_CONV:.0f} s', color='#ff6b6b', fontsize=9)

# Animated: growing error curves
error_lines = [
    ax_er.plot([], [], '-', color=COLORS[i], lw=1.8, label=LABELS[i])[0]
    for i in range(N)
]

ax_er.legend(loc='upper right',
             facecolor='#21262d', edgecolor=GRID_COL, labelcolor=TEXT_COL,
             fontsize=9)

# ── Animation update function ────────────────────────────────────────────────
anim_frames = list(range(0, n_frames, SKIP))


def update(frame_idx: int):
    k     = anim_frames[frame_idx]
    t_now = sim.t_hist[k]
    p_now = sim.p_hist[k]            # (N, 2)

    # 1. Trajectory paths
    for i in range(N):
        traj_lines[i].set_data(sim.p_hist[:k+1, i, 0],
                               sim.p_hist[:k+1, i, 1])

    # 2. Current position markers + labels
    for i in range(N):
        leader_dots[i].set_data([p_now[i, 0]], [p_now[i, 1]])
        leader_lbls[i].set_position((p_now[i, 0], p_now[i, 1] + 0.28))

    # 3. Live communication edges  V → Li (current position)
    for i in range(N):
        comm_arrows[i].xy    = (p_now[i, 0], p_now[i, 1])   # arrow head
        comm_arrows[i].xyann = (0.0, 0.0)                    # arrow tail (V)

    # 4. Growing error curves
    for i in range(N):
        error_lines[i].set_data(sim.t_hist[:k+1], sim.xi_hist[:k+1, i])

    # 5. Time label
    time_text.set_text(f't = {t_now:5.2f} s')

    # 6. Convergence badge
    if t_now >= T_CONV:
        conv_badge.set_text('✓ Converged')
    else:
        conv_badge.set_text('')

    artists = (traj_lines + leader_dots + leader_lbls +
               error_lines + [time_text, conv_badge])
    return artists


# ── Build and save animation as GIF ─────────────────────────────────────────
anim = FuncAnimation(
    fig,
    update,
    frames=len(anim_frames),
    interval=50,        # 50 ms per frame → ~20 fps
    blit=False,         # blit=False keeps annotation arrows working correctly
    repeat=True,
)

output_path = 'animation/formation_error.gif'
print(f"Saving GIF to {output_path}  ({len(anim_frames)} frames) …")
anim.save(output_path, writer='pillow', fps=20, dpi=110)
print("Done →", output_path)
