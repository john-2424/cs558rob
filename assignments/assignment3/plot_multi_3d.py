import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import sys


def load_obs(fname):
    p = np.fromfile(fname)
    return p.reshape(len(p) // 3, 3)


def load_path(fname):
    p = np.loadtxt(fname)
    return p.reshape(-1, 3)


def cost(path):
    if len(path) < 2:
        return 0.0
    return np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))


if len(sys.argv) < 4:
    print("Usage:")
    print("python plot_multi_3d.py <obs_file> <out_file> <path1.txt> <path2.txt> ... <path5.txt>")
    sys.exit(1)

obs_file = sys.argv[1]
out_file = sys.argv[2]
path_files = sys.argv[3:]

obs = load_obs(obs_file)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# obstacle cloud
ax.scatter(
    obs[:, 0], obs[:, 1], obs[:, 2],
    s=2, c='gray', alpha=0.15, label='Obstacle cloud'
)

for i, pf in enumerate(path_files):
    p = load_path(pf)
    c = cost(p)

    ax.plot(
        p[:, 0], p[:, 1], p[:, 2],
        linewidth=2.2,
        label=f"Run {i+1}: {c:.2f}"
    )

    if i == 0:
        ax.scatter(
            p[0, 0], p[0, 1], p[0, 2],
            c='green', s=60, label='Start'
        )
        ax.scatter(
            p[-1, 0], p[-1, 1], p[-1, 2],
            c='brown', s=60, label='Goal'
        )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Five MPNet runs for a fixed start-goal pair")
ax.legend()
plt.tight_layout()
plt.savefig(out_file, dpi=250)
plt.close()

print(f"Saved figure to: {out_file}")
for i, pf in enumerate(path_files):
    p = load_path(pf)
    print(f"Run {i+1} cost: {cost(p):.6f}  file: {pf}")