import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import sys


def load_expert(fname):
    p = np.fromfile(fname)
    return p.reshape(len(p) // 3, 3)


def load_mpnet(fname):
    p = np.loadtxt(fname)
    return p.reshape(-1, 3)


def load_obs(fname):
    p = np.fromfile(fname)
    return p.reshape(len(p) // 3, 3)


def cost(path):
    if len(path) < 2:
        return 0.0
    return np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))


if len(sys.argv) != 5:
    print("Usage:")
    print("python plot_compare_3d.py <obs_file> <expert_file> <mpnet_file> <out_file>")
    sys.exit(1)

obs_file, expert_file, mpnet_file, out_file = sys.argv[1:5]

obs = load_obs(obs_file)
expert = load_expert(expert_file)
mpnet = load_mpnet(mpnet_file)

c_rrt = cost(expert)
c_mp = cost(mpnet)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# obstacle cloud
ax.scatter(
    obs[:, 0], obs[:, 1], obs[:, 2],
    s=2, c='gray', alpha=0.15, label='Obstacle cloud'
)

# expert path (RRT*)
ax.plot(
    expert[:, 0], expert[:, 1], expert[:, 2],
    'b-', linewidth=2.5, label=f"RRT*: {c_rrt:.2f}"
)

# MPNet path
ax.plot(
    mpnet[:, 0], mpnet[:, 1], mpnet[:, 2],
    'r-', linewidth=2.5, label=f"MPNet: {c_mp:.2f}"
)

# start and goal
ax.scatter(
    expert[0, 0], expert[0, 1], expert[0, 2],
    c='green', s=60, label='Start'
)
ax.scatter(
    expert[-1, 0], expert[-1, 1], expert[-1, 2],
    c='brown', s=60, label='Goal'
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(f"MPNet (Red): cost={c_mp:.2f} ; RRT* (Blue): cost={c_rrt:.2f}")
ax.legend()
plt.tight_layout()
plt.savefig(out_file, dpi=250)
plt.close()

print(f"Saved figure to: {out_file}")
print(f"MPNet cost: {c_mp:.6f}")
print(f"RRT* cost: {c_rrt:.6f}")