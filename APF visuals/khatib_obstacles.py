import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


# ----------------------------------------------------
# Base obstacle class
# ----------------------------------------------------
class Obstacle:
    def distance(self, x, y, z):
        raise NotImplementedError


# ----------------------------------------------------
# Sphere
# ----------------------------------------------------
class SphereObstacle(Obstacle):
    def __init__(self, center, radius):
        self.cx, self.cy, self.cz = center
        self.r = radius

    def distance(self, x, y, z):
        return np.sqrt((x - self.cx)**2 +
                       (y - self.cy)**2 +
                       (z - self.cz)**2) - self.r


# ----------------------------------------------------
# Axis-aligned box
# ----------------------------------------------------
class BoxObstacle(Obstacle):
    def __init__(self, center, size):
        self.cx, self.cy, self.cz = center
        self.sx, self.sy, self.sz = size

    def distance(self, x, y, z):
        dx = np.abs(x - self.cx) - self.sx/2
        dy = np.abs(y - self.cy) - self.sy/2
        dz = np.abs(z - self.cz) - self.sz/2
        dx = np.maximum(dx, 0)
        dy = np.maximum(dy, 0)
        dz = np.maximum(dz, 0)
        return np.sqrt(dx*dx + dy*dy + dz*dz)


# ----------------------------------------------------
# Triangle obstacle (single triangular face)
# ----------------------------------------------------
class TriangleObstacle(Obstacle):
    def __init__(self, p1, p2, p3, thickness=0.02):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)
        self.thickness = thickness

        v1 = self.p2 - self.p1
        v2 = self.p3 - self.p1
        n = np.cross(v1, v2)
        self.normal = n / np.linalg.norm(n)

    def distance(self, x, y, z):
        pts = np.stack([x, y, z], axis=-1)

        p1 = self.p1
        n = self.normal
        eps = self.thickness

        # Signed distance to plane
        d_plane = np.sum((pts - p1) * n, axis=-1)

        # Projection onto plane
        proj = pts - d_plane[..., None] * n

        # Barycentric inside test
        v0 = self.p3 - self.p1
        v1 = self.p2 - self.p1
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        denom = d00 * d11 - d01 * d01

        v2 = proj - self.p1
        d20 = np.sum(v2 * v0, axis=-1)
        d21 = np.sum(v2 * v1, axis=-1)

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w

        inside = (u >= 0) & (v >= 0) & (w >= 0)

        # If inside triangle, signed distance = plane distance
        d = np.where(inside, d_plane, np.inf)

        # Segment distance helper
        def segment_dist(p, a, b):
            a = a.reshape(1, 1, 1, 3)
            b = b.reshape(1, 1, 1, 3)
            ab = b - a
            t = np.sum((p - a) * ab, axis=-1) / np.sum(ab * ab)
            t = np.clip(t, 0, 1)[..., None]
            closest = a + t * ab
            return np.linalg.norm(p - closest, axis=-1)

        d1 = segment_dist(pts, self.p1, self.p2)
        d2 = segment_dist(pts, self.p2, self.p3)
        d3 = segment_dist(pts, self.p3, self.p1)

        unsigned = np.minimum(d, np.minimum(d1, np.minimum(d2, d3)))

        # Signed distance: negative inside the thin slab
        signed = np.where(np.abs(d_plane) <= eps, -unsigned, unsigned)

        return signed
# ----------------------------------------------------
# Arbitrary non-convex shape via voxel SDF
# ----------------------------------------------------
class NonConvexObstacle(Obstacle):
    def __init__(self, sdf_grid, bounds):
        self.sdf = sdf_grid
        self.bounds = bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)

    def distance(self, x, y, z):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        nx, ny, nz = self.sdf.shape

        # Normalize coordinates to grid index space
        gx = (x - xmin) / (xmax - xmin) * (nx - 1)
        gy = (y - ymin) / (ymax - ymin) * (ny - 1)
        gz = (z - zmin) / (zmax - zmin) * (nz - 1)

        # Trilinear interpolation
        gx0 = np.floor(gx).astype(int)
        gy0 = np.floor(gy).astype(int)
        gz0 = np.floor(gz).astype(int)
        gx1 = np.clip(gx0 + 1, 0, nx - 1)
        gy1 = np.clip(gy0 + 1, 0, ny - 1)
        gz1 = np.clip(gz0 + 1, 0, nz - 1)

        xd = gx - gx0
        yd = gy - gy0
        zd = gz - gz0

        c000 = self.sdf[gx0, gy0, gz0]
        c100 = self.sdf[gx1, gy0, gz0]
        c010 = self.sdf[gx0, gy1, gz0]
        c110 = self.sdf[gx1, gy1, gz0]
        c001 = self.sdf[gx0, gy0, gz1]
        c101 = self.sdf[gx1, gy0, gz1]
        c011 = self.sdf[gx0, gy1, gz1]
        c111 = self.sdf[gx1, gy1, gz1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        return c0 * (1 - zd) + c1 * zd


# ----------------------------------------------------
# Unified renderer
# ----------------------------------------------------
def plot_all(obstacles, p0=0.4, bounds=(-2, 2), resolution=80):
    xmin, xmax = bounds
    ymin, ymax = bounds
    zmin, zmax = bounds

    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    zs = np.linspace(zmin, zmax, resolution)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = [(0.2, 0.2, 0.8), (0.2, 0.6, 0.2), (0.8, 0.2, 0.2), (0.6, 0.2, 0.6)]

    for idx, obs in enumerate(obstacles):
        D = obs.distance(X, Y, Z)

        # Obstacle surface
        verts, faces, _, _ = marching_cubes(D, level=0)
        scale = (xmax - xmin) / resolution
        verts = verts * scale + xmin
        mesh = Poly3DCollection(verts[faces], alpha=1.0)
        mesh.set_facecolor(colors[idx % len(colors)])
        ax.add_collection3d(mesh)

        # Influence region
        verts2, faces2, _, _ = marching_cubes(D, level=p0)
        verts2 = verts2 * scale + xmin
        shell = Poly3DCollection(verts2[faces2], alpha=0.25)
        shell.set_facecolor(colors[idx % len(colors)])
        ax.add_collection3d(shell)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Multiple Obstacles with APF Influence Regions")

    plt.show()


# ----------------------------------------------------
# Example usage
# ----------------------------------------------------
if __name__ == "__main__":
    sphere = SphereObstacle(center=(0, 0, 0), radius=0.5)
    box = BoxObstacle(center=(1.0, 0.5, 0.0), size=(0.8, 0.4, 0.6))
    box2 = BoxObstacle(center=(-0.5, -0.5, -0.5), size=(0.6, 0.6, 0.6))
    # tri = TriangleObstacle(p1=(-0.5, -0.5, 0.2),
    #                        p2=(-1.0, 0.5, 0.2),
    #                        p3=(-0.2, 0.3, 1.0))

    # Non-convex SDF: a torus-like shape
    nx = ny = nz = 64
    xs = np.linspace(-1, 1, nx)
    ys = np.linspace(-1, 1, ny)
    zs = np.linspace(-1, 1, nz)
    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')
    sdf = np.sqrt((np.sqrt(Xg**2 + Yg**2) - 0.5)**2 + Zg**2) - 0.2
    # nonconvex = NonConvexObstacle(sdf, bounds=(-1, 1, -1, 1, -1, 1))

    plot_all([sphere, box, box2], p0=0.1, bounds=(-2, 2), resolution=80)