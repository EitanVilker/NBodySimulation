import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits import mplot3d

# Enables 3D projection
fig = plt.figure()
ax = plt.axes(projection="3d")

def update(frame):

    input_file_name = str(frame) + '.txt'
    print(input_file_name)
    input_file = open(input_file_name, "r")

    n = 16 * 16 * 16


    for i in range(n):

        current_line = input_file.readline()
        point = current_line.split(',')

        x = float(point[0])
        y = float(point[1])
        z = float(point[2])

        ax.plot3D([x], [y], [z], marker='o', markersize='3', color='red')
    
    input_file.close()

# Create graph
# ani = animation.FuncAnimation(fig, update, frames=10)

update(1)

plt.show()

