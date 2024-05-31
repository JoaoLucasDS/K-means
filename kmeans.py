import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def kMeans(dataBase, K, D):
    N = (len(dataBase))

    print(dataBase.shape)
    means = np.zeros((K, D))

    # Select K unique random initial points
    unique_points = set()
    while len(unique_points) < K:
        i = np.random.choice(N)
        unique_points.add(tuple(dataBase[i]))
    means = np.array(list(unique_points))

    clusterId = np.zeros(N)
    maxIter = 100

    fig, ax = plt.subplots()

    def update_plot(iter):
        nonlocal means, clusterId

        ax.clear()

        oldClusterId = clusterId.copy()  # Define oldClusterId here

        # Assign each point to the nearest cluster center
        for pointIndex in range(N):
            dists = np.linalg.norm(dataBase[pointIndex] - means, axis=1)
            clusterId[pointIndex] = np.argmin(dists)

        # Update cluster centers
        for clusterIndex in range(K):
            if np.any(clusterId == clusterIndex):
                means[clusterIndex] = dataBase[clusterId == clusterIndex].mean(axis=0)

        # Plot the results
        for i in range(K):
            ax.scatter(means[i, 1], means[i, 0], s=100, c='black', marker='*')
        ax.scatter(dataBase[:, 1], dataBase[:, 0], c=clusterId, cmap='viridis')

        ax.set_title(f'Iteration {iter+1}')

        # Dynamically set the limits of the plot
        x_min, x_max = np.min(dataBase[:, 1]), np.max(dataBase[:, 1])
        y_min, y_max = np.min(dataBase[:, 0]), np.max(dataBase[:, 0])
        x_margin = (x_max - x_min) * 0.1  # Add 10% margin
        y_margin = (y_max - y_min) * 0.1  # Add 10% margin
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Check for convergence
        if iter < maxIter - 1 and np.all(oldClusterId == clusterId):
            print("Número de iterações:", iter)
            ani.event_source.stop()

    ani = FuncAnimation(fig, update_plot, frames=maxIter, repeat=False)
    plt.show()
