import numpy as np
import matplotlib.pyplot as plt
import sys

obs_file = sys.argv[1]
out_file = sys.argv[2]
path_files = sys.argv[3:]

obs = np.fromfile(obs_file).reshape(-1,2)
plt.figure(figsize=(6,6))
plt.scatter(obs[:,0], obs[:,1], s=5, c='gray', alpha=0.4, label='Obstacle cloud')

for i, pf in enumerate(path_files):
    p = np.loadtxt(pf).reshape(-1,2)
    plt.plot(p[:,0], p[:,1], linewidth=2, label=f'Run {i+1}')
    if i == 0:
        plt.scatter(p[0,0], p[0,1], c='green', s=50, label='Start')
        plt.scatter(p[-1,0], p[-1,1], c='brown', s=50, label='Goal')

plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.savefig(out_file, dpi=200)