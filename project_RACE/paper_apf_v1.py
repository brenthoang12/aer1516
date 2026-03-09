import numpy as np
import matplotlib.pyplot as plt

from crazyflow.control import Control
from crazyflow.sim import Sim


# ------------------------------------------------------------
# Virtual leader trajectory
# ------------------------------------------------------------
def virtual_leader(t, p0=np.array([0.0, 0.0, 1.0]), v_ref=np.array([0.2, 0.0, 0.0])):
    p_v = p0 + v_ref * t
    return p_v, v_ref


# ------------------------------------------------------------
# Cube formation offsets for 8 leaders
# ------------------------------------------------------------
def cube_offsets(edge=1.0):
    hs = edge / 2.0
    return np.array([
        [-hs, -hs, -hs],
        [-hs, -hs,  hs],
        [-hs,  hs, -hs],
        [-hs,  hs,  hs],
        [ hs, -hs, -hs],
        [ hs, -hs,  hs],
        [ hs,  hs, -hs],
        [ hs,  hs,  hs],
    ])


# ------------------------------------------------------------
# Adjacency matrix (approximate Fig. 3)
# Followers: 0..7, Leaders: 8..15
# ------------------------------------------------------------
def build_adjacency():
    N = 16
    A = np.zeros((N, N))

    # Leaders: chain + reverse chain
    for i in range(8, 15):
        A[i, i + 1] = 1.0
    for i in range(9, 16):
        A[i, i - 1] = 1.0

    # Followers: each connected to two leaders
    for k in range(0, 8):
        A[k, 8 + (k % 4)] = 1.0
        A[k, 8 + ((k + 1) % 4)] = 1.0

    return A


# ------------------------------------------------------------
# Compute ξ and ζ (paper definitions)
# ------------------------------------------------------------
def compute_xi_zeta(p, v, leader_idx, follower_idx, h_leaders, A):
    N = p.shape[0]
    h = np.zeros((N, 3))
    h[leader_idx] = h_leaders

    xi = np.zeros_like(p)
    zeta = np.zeros_like(p)

    # Leaders
    for i in leader_idx:
        for j in range(N):
            if A[i, j] != 0:
                d_ij = h[i] - h[j]
                xi[i] += A[i, j] * (p[i] - p[j] - d_ij)
                zeta[i] += A[i, j] * (v[i] - v[j])

    # Followers
    for k in follower_idx:
        for j in range(N):
            if A[k, j] != 0:
                xi[k] += A[k, j] * (p[k] - p[j])
                zeta[k] += A[k, j] * (v[k] - v[j])

    return xi, zeta


# ------------------------------------------------------------
# Mild repulsion (unchanged)
# ------------------------------------------------------------
def all_pair_repulsion(p, d_safe=0.2, k_rep=0.05):
    N = p.shape[0]
    u_rep = np.zeros_like(p)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = p[i] - p[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue
            if dist < d_safe:
                dir_ij = diff / dist
                mag = k_rep * (1.0 / dist - 1.0 / d_safe)
                u_rep[i] += mag * dir_ij
    return u_rep


# ------------------------------------------------------------
# FIXED-TIME CONTROLLER (unchanged)
# ------------------------------------------------------------
def fixed_time_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders,
                       a=1.0, b=0.8, c=0.5, h=5.0, q=0.5,
                       d_safe=0.2, k_rep=0.05):
    u = np.zeros_like(p)

    # Leaders
    for i_local, i in enumerate(leader_idx):
        p_des = p_v + h_leaders[i_local]
        v_des = v_v
        e = p[i] - p_des
        de = v[i] - v_des
        s = e + (1.0 / h) * de
        u[i] = -a * s - b * np.power(np.abs(s), q) * np.sign(s) - c * de

    # Followers
    p_leaders = p[leader_idx]
    v_leaders = v[leader_idx]
    p_ref = np.mean(p_leaders, axis=0)
    v_ref = np.mean(v_leaders, axis=0)

    for k in follower_idx:
        e = p[k] - p_ref
        de = v[k] - v_ref
        s = e + (1.0 / h) * de
        u[k] = -a * s - b * np.power(np.abs(s), q) * np.sign(s) - c * de

    u += all_pair_repulsion(p, d_safe=d_safe, k_rep=k_rep)
    return u


# ------------------------------------------------------------
# THEOREM 1 BENCHMARK CONTROLLER (correct linear-system version)
#   (Γ - A) u = -k_p ξ - k_v ζ
# ------------------------------------------------------------
def benchmark_control(p, v, leader_idx, follower_idx, h_leaders, A,
                      k_p=0.05, k_v=0.05, d_safe=0.2, k_rep=0.05):
    N = p.shape[0]
    xi, zeta = compute_xi_zeta(p, v, leader_idx, follower_idx, h_leaders, A)

    gamma = A.sum(axis=1)  # shape (N,)
    M = np.diag(gamma) - A  # (Γ - A)

    rhs = -(k_p * xi + k_v * zeta)  # shape (N,3)
    u = np.zeros_like(p)

    # solve per coordinate
    for dim in range(3):
        u[:, dim] = np.linalg.solve(M, rhs[:, dim])

    # keep your mild repulsion
    u += all_pair_repulsion(p, d_safe=d_safe, k_rep=k_rep)
    return u


# ------------------------------------------------------------
# PAPER-STYLE ERROR (per-agent ξ norms)
# ------------------------------------------------------------
def compute_errors(p, v, leader_idx, follower_idx, h_leaders, A):
    xi, zeta = compute_xi_zeta(p, v, leader_idx, follower_idx, h_leaders, A)
    return np.linalg.norm(xi[leader_idx], axis=1), np.linalg.norm(xi[follower_idx], axis=1)


# ------------------------------------------------------------
# SIMULATION (Crazyflow + double integrator)
# ------------------------------------------------------------
def run_sim(use_fixed_time=True, T=30.0, v_ref=np.array([0.2, 0.0, 0.0])):
    N = 16
    leader_idx = np.arange(8, 16)
    follower_idx = np.arange(0, 8)

    p = np.zeros((N, 3))
    v = np.zeros((N, 3))

    rng = np.random.default_rng(0)
    p[leader_idx] = rng.uniform([-1, -1, 0.5], [1, 1, 1.5], size=(8, 3))
    p[follower_idx] = rng.uniform([-2, -2, 0], [2, 2, 0.5], size=(8, 3))

    h_leaders = cube_offsets(edge=1.0)
    A = build_adjacency()

    dt = 1.0 / 50.0
    steps = int(T * 50.0)

    xiL_hist = []
    xiF_hist = []
    t_hist = []

    sim = Sim(n_drones=N, control=Control.state)
    sim.reset()

    for k in range(steps):
        t = k * dt
        p_v, v_v = virtual_leader(t)

        if use_fixed_time:
            u = fixed_time_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders)
        else:
            u = benchmark_control(p, v, leader_idx, follower_idx, h_leaders, A)

        v = v + u * dt
        p = p + v * dt

        xiL, xiF = compute_errors(p, v, leader_idx, follower_idx, h_leaders, A)
        xiL_hist.append(xiL)
        xiF_hist.append(xiF)
        t_hist.append(t)

        states = sim.data.states.replace(
            pos=sim.data.states.pos.at[0].set(p),
            vel=sim.data.states.vel.at[0].set(v),
        )
        sim.data = sim.data.replace(states=states)
        sim.step(1)

        if k % 5 == 0:  # <--- Option C
            sim.render()

    sim.close()
    return np.array(t_hist), np.array(xiL_hist), np.array(xiF_hist)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    t_ft, L_ft, F_ft = run_sim(use_fixed_time=True, T=40.0)
    t_bm, L_bm, F_bm = run_sim(use_fixed_time=False, T=40.0)

    plt.figure(figsize=(12, 8))

    # Fixed-time leaders
    plt.subplot(2, 2, 1)
    for i in range(8):
        plt.plot(t_ft, L_ft[:, i], label=f"L{i+1}")
    plt.title("Fixed-time (Theorem 2 approx): Leader Errors")
    plt.xlabel("Time [s]")
    plt.ylabel("||ξ_i||")
    plt.grid(True)

    # Fixed-time followers
    plt.subplot(2, 2, 2)
    for k in range(8):
        plt.plot(t_ft, F_ft[:, k], label=f"F{k+1}")
    plt.title("Fixed-time (Theorem 2 approx): Follower Errors")
    plt.xlabel("Time [s]")
    plt.ylabel("||ξ_k||")
    plt.grid(True)

    # Benchmark leaders
    plt.subplot(2, 2, 3)
    for i in range(8):
        plt.plot(t_bm, L_bm[:, i], label=f"L{i+1}")
    plt.title("Benchmark (Theorem 1): Leader Errors")
    plt.xlabel("Time [s]")
    plt.ylabel("||ξ_i||")
    plt.grid(True)

    # Benchmark followers
    plt.subplot(2, 2, 4)
    for k in range(8):
        plt.plot(t_bm, F_bm[:, k], label=f"F{k+1}")
    plt.title("Benchmark (Theorem 1): Follower Errors")
    plt.xlabel("Time [s]")
    plt.ylabel("||ξ_k||")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


"""
import numpy as np
import matplotlib.pyplot as plt

from crazyflow.control import Control
from crazyflow.sim import Sim


# ------------------------------------------------------------
# Virtual leader trajectory
# ------------------------------------------------------------
def virtual_leader(t, p0=np.array([0.0, 0.0, 1.0]), v_ref=np.array([0.2, 0.0, 0.0])):
    p_v = p0 + v_ref * t
    return p_v, v_ref


# ------------------------------------------------------------
# Cube formation offsets for 8 leaders
# ------------------------------------------------------------
def cube_offsets(edge=1.0):
    hs = edge / 2.0
    return np.array([
        [-hs, -hs, -hs],
        [-hs, -hs,  hs],
        [-hs,  hs, -hs],
        [-hs,  hs,  hs],
        [ hs, -hs, -hs],
        [ hs, -hs,  hs],
        [ hs,  hs, -hs],
        [ hs,  hs,  hs],
    ])


# ------------------------------------------------------------
# Adjacency matrix (approximate Fig. 3)
# Followers: 0..7, Leaders: 8..15
# ------------------------------------------------------------
def build_adjacency():
    N = 16
    A = np.zeros((N, N))

    # Leaders: chain + reverse chain
    for i in range(8, 15):
        A[i, i + 1] = 1.0
    for i in range(9, 16):
        A[i, i - 1] = 1.0

    # Followers: each connected to two leaders
    for k in range(0, 8):
        A[k, 8 + (k % 4)] = 1.0
        A[k, 8 + ((k + 1) % 4)] = 1.0

    return A


# ------------------------------------------------------------
# Compute ξ and ζ (paper definitions)
# ------------------------------------------------------------
def compute_xi_zeta(p, v, leader_idx, follower_idx, h_leaders, A):
    N = p.shape[0]
    h = np.zeros((N, 3))
    h[leader_idx] = h_leaders

    xi = np.zeros_like(p)
    zeta = np.zeros_like(p)

    # Leaders
    for i in leader_idx:
        for j in range(N):
            if A[i, j] != 0:
                d_ij = h[i] - h[j]
                xi[i] += A[i, j] * (p[i] - p[j] - d_ij)
                zeta[i] += A[i, j] * (v[i] - v[j])

    # Followers
    for k in follower_idx:
        for j in range(N):
            if A[k, j] != 0:
                xi[k] += A[k, j] * (p[k] - p[j])
                zeta[k] += A[k, j] * (v[k] - v[j])

    return xi, zeta


# ------------------------------------------------------------
# Mild repulsion (unchanged)
# ------------------------------------------------------------
def all_pair_repulsion(p, d_safe=0.2, k_rep=0.05):
    N = p.shape[0]
    u_rep = np.zeros_like(p)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = p[i] - p[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue
            if dist < d_safe:
                dir_ij = diff / dist
                mag = k_rep * (1.0 / dist - 1.0 / d_safe)
                u_rep[i] += mag * dir_ij
    return u_rep


# ------------------------------------------------------------
# FIXED-TIME CONTROLLER (unchanged)
# ------------------------------------------------------------
def fixed_time_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders,
                       a=1.0, b=0.8, c=0.5, h=5.0, q=0.5,
                       d_safe=0.2, k_rep=0.05):
    u = np.zeros_like(p)

    # Leaders
    for i_local, i in enumerate(leader_idx):
        p_des = p_v + h_leaders[i_local]
        v_des = v_v
        e = p[i] - p_des
        de = v[i] - v_des
        s = e + (1.0 / h) * de
        u[i] = -a * s - b * np.power(np.abs(s), q) * np.sign(s) - c * de

    # Followers
    p_leaders = p[leader_idx]
    v_leaders = v[leader_idx]
    p_ref = np.mean(p_leaders, axis=0)
    v_ref = np.mean(v_leaders, axis=0)

    for k in follower_idx:
        e = p[k] - p_ref
        de = v[k] - v_ref
        s = e + (1.0 / h) * de
        u[k] = -a * s - b * np.power(np.abs(s), q) * np.sign(s) - c * de

    u += all_pair_repulsion(p, d_safe=d_safe, k_rep=k_rep)
    return u


# ------------------------------------------------------------
# THEOREM 1 BENCHMARK CONTROLLER (correct version)
# ------------------------------------------------------------
def benchmark_control(p, v, leader_idx, follower_idx, h_leaders, A, u_prev,
                      k_p=0.05, k_v=0.05):
    N = p.shape[0]
    u = np.zeros_like(p)

    xi, zeta = compute_xi_zeta(p, v, leader_idx, follower_idx, h_leaders, A)
    gamma = A.sum(axis=1)

    for i in range(N):
        if gamma[i] <= 1e-8:
            continue

        # neighbor accelerations approx = previous-step u_j
        sum_aij_uj = (A[i, :, None] * u_prev).sum(axis=0)

        u[i] = -(1.0 / gamma[i]) * (k_p * xi[i] + k_v * zeta[i] - sum_aij_uj)

    return u


# ------------------------------------------------------------
# PAPER-STYLE ERROR (per-agent ξ norms)
# ------------------------------------------------------------
def compute_errors(p, v, leader_idx, follower_idx, h_leaders, A):
    xi, zeta = compute_xi_zeta(p, v, leader_idx, follower_idx, h_leaders, A)
    return np.linalg.norm(xi[leader_idx], axis=1), np.linalg.norm(xi[follower_idx], axis=1)


# ------------------------------------------------------------
# SIMULATION (Crazyflow + double integrator)
# ------------------------------------------------------------
def run_sim(use_fixed_time=True, T=30.0, v_ref=np.array([0.2, 0.0, 0.0])):
    N = 16
    leader_idx = np.arange(8, 16)
    follower_idx = np.arange(0, 8)

    p = np.zeros((N, 3))
    v = np.zeros((N, 3))

    rng = np.random.default_rng(0)
    p[leader_idx] = rng.uniform([-1, -1, 0.5], [1, 1, 1.5], size=(8, 3))
    p[follower_idx] = rng.uniform([-2, -2, 0], [2, 2, 0.5], size=(8, 3))

    h_leaders = cube_offsets(edge=1.0)
    A = build_adjacency()

    dt = 1.0 / 50.0
    steps = int(T * 50.0)

    xiL_hist = []
    xiF_hist = []
    t_hist = []

    u_prev = np.zeros_like(p)

    sim = Sim(n_drones=N, control=Control.state)
    sim.reset()

    for k in range(steps):
        t = k * dt
        p_v, v_v = virtual_leader(t)

        if use_fixed_time:
            u = fixed_time_control(p, v, p_v, v_v, leader_idx, follower_idx, h_leaders)
        else:
            u = benchmark_control(p, v, leader_idx, follower_idx, h_leaders, A, u_prev)

        v = v + u * dt
        p = p + v * dt
        u_prev = u.copy()

        xiL, xiF = compute_errors(p, v, leader_idx, follower_idx, h_leaders, A)
        xiL_hist.append(xiL)
        xiF_hist.append(xiF)
        t_hist.append(t)

        states = sim.data.states.replace(
            pos=sim.data.states.pos.at[0].set(p),
            vel=sim.data.states.vel.at[0].set(v),
        )
        sim.data = sim.data.replace(states=states)
        sim.step(1)

        if k % 5 == 0: # <--- Option C 
            sim.render()

    sim.close()
    return np.array(t_hist), np.array(xiL_hist), np.array(xiF_hist)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    t_ft, L_ft, F_ft = run_sim(use_fixed_time=True, T=40.0)
    t_bm, L_bm, F_bm = run_sim(use_fixed_time=False, T=40.0)

    plt.figure(figsize=(12, 8))

    # Fixed-time leaders
    plt.subplot(2, 2, 1)
    for i in range(8):
        plt.plot(t_ft, L_ft[:, i], label=f"L{i+1}")
    plt.title("Fixed-time (Theorem 2 approx): Leader Errors")
    plt.xlabel("Time [s]")
    plt.ylabel("||ξ_i||")
    plt.grid(True)

    # Fixed-time followers
    plt.subplot(2, 2, 2)
    for k in range(8):
        plt.plot(t_ft, F_ft[:, k], label=f"F{k+1}")
    plt.title("Fixed-time (Theorem 2 approx): Follower Errors")
    plt.xlabel("Time [s]")
    plt.ylabel("||ξ_k||")
    plt.grid(True)

    # Benchmark leaders
    plt.subplot(2, 2, 3)
    for i in range(8):
        plt.plot(t_bm, L_bm[:, i], label=f"L{i+1}")
    plt.title("Benchmark (Theorem 1): Leader Errors")
    plt.xlabel("Time [s]")
    plt.ylabel("||ξ_i||")
    plt.grid(True)

    # Benchmark followers
    plt.subplot(2, 2, 4)
    for k in range(8):
        plt.plot(t_bm, F_bm[:, k], label=f"F{k+1}")
    plt.title("Benchmark (Theorem 1): Follower Errors")
    plt.xlabel("Time [s]")
    plt.ylabel("||ξ_k||")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
"""