import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
def debug_visualize(points, voxels, objects, filepath):
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots(1, 1, figsize=(76.8, 76.8))
    min_x, max_x, min_y, max_y = -76.8, 76.8, -76.8, 76.8
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    # voxels
    voxel_size = 0.8  # TODO: Change this if we want to keep the visualize function
    points = np.array([p for p in points if -1 <= p[2] <= 3])
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            voxel_x_idx = int((x+76.8) // voxel_size)
            voxel_y_idx = int((y+76.8) // voxel_size)
            try:
                if voxels[voxel_y_idx+1, voxel_x_idx+1] > 0.5:
                    ax.add_patch(Rectangle((x, y), voxel_size, voxel_size, color=(0, 1, 0, 0.3)))
            except:
                print(voxel_y_idx, voxel_x_idx, y, x)
                
            y += voxel_size
        x += voxel_size
    # points
    ax.scatter(points[:, 0], points[:, 1], s=10)
    # object centers
    ax.scatter(objects[:, 0], objects[:, 1], s=150, c='red')
    fig.savefig(filepath)
import numpy as np
points = np.load('points.npy')
voxels = np.load('hotspots.npy')
objects = np.load('gt_bboxes.npy')
filepath='vis.jpg'
debug_visualize(points,voxels, objects, filepath)
