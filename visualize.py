import matplotlib.pyplot as plt
import numpy as np

from window import convert_map_to_numpy

def visualize_policy(contents, policy, mode="window"):
    map = convert_map_to_numpy(contents, include_owner_channel=True)

    fig = plt.figure(dpi=150, figsize=(10, 10))
    ax = fig.gca()

    owner = map[:, :, 6]

    production = map[:, :, 3:6]
    production[:, :, 2][owner==0] += 4
    production[:, :, 0][owner==1] += 4
    production[:, :, 1][owner==2] += 4
    production /= 16

    strength = map[:, :, :3] / 255


    ax.matshow(production)

    ax.scatter(*tuple(reversed(np.where((policy == 0) & (owner == 1)))),
        marker="^", color="red", edgecolor="white",
        s=(strength[policy == 0] + 0.2) * 100
    )

    ax.scatter(*tuple(reversed(np.where((policy == 1) & (owner == 1)))),
        marker=">", color="red", edgecolor="white",
        s=(strength[policy == 1] + 0.2) * 100
    )

    ax.scatter(*tuple(reversed(np.where((policy == 2) & (owner == 1)))),
        marker="v", color="red", edgecolor="white",
        s=(strength[policy == 2] + 0.2) * 100
    )

    ax.scatter(*tuple(reversed(np.where((policy == 3) & (owner == 1)))),
        marker="<", color="red", edgecolor="white",
        s=(strength[policy == 3] + 0.2) * 100
    )

    ax.scatter(*tuple(reversed(np.where((policy == 4) & (owner == 1)))),
        marker="o", color="red", edgecolor="white",
        s=(strength[policy == 4] + 0.2) * 100
    )

    fig.show()
