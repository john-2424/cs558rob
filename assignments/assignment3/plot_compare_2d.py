import numpy as np
import matplotlib.pyplot as plt
import sys

def load_expert(fname):
    p = np.fromfile(fname)
    return p.reshape(len(p)//2, 2)

def load_mpnet(fname):
    p = np.loadtxt(fname)
    return p.reshape(-1, 2)

def cost(path):
    return np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))

obs_file, expert_file, mpnet_file, out_file = sys.argv[1:5]

obs = np.fromfile(obs_file).reshape(-1, 2)
expert = load_expert(expert_file)
mpnet = load_mpnet(mpnet_file)

c_rrt = cost(expert)
c_mp = cost(mpnet)

plt.figure(figsize=(6,6))
plt.scatter(obs[:,0], obs[:,1], s=5, c='gray', alpha=0.35)
plt.plot(mpnet[:,0], mpnet[:,1], 'r-', linewidth=2, label=f"MPNet: {c_mp:.1f}")
plt.plot(expert[:,0], expert[:,1], 'b-', linewidth=2, label=f"RRT*: {c_rrt:.1f}")
plt.scatter(expert[0,0], expert[0,1], c='green', s=40)
plt.scatter(expert[-1,0], expert[-1,1], c='brown', s=40)
plt.axis('equal')
plt.legend()
plt.title(f"MPNet(Red): cost={c_mp:.1f} ; RRT*(Blue): cost={c_rrt:.1f}")
plt.tight_layout()
plt.savefig(out_file, dpi=200)

print(f"MPNet cost: {c_mp:.3f}")
print(f"RRT* cost: {c_rrt:.3f}")