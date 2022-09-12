"""
Based on:
Jos Stam Real-Time Fluid Dynamics       https://www.researchgate.net/publication/2560062_Real-Time_Fluid_Dynamics_for_Games       
Mikeash Fluid Simulation for Dummies    https://www.mikeash.com/pyblog/fluid-simulation-for-dummies.html
"""


import random
import numpy as np
from solver import Fluid
global test

def main():
    import matplotlib.pyplot as plt
    from matplotlib import animation

    N = 64
    dt = 0.1
    it = 10  # iteration
    diff = 0.0000
    visc = 0.0000

    fluid = Fluid(N, dt, it, diff, visc)

    def update_im(i):
        # Some math to make the source of density spin at the center of the map
        fluid.density[int(N / 2 - 2):int(N / 2), int(N / 2 - 2):int(N / 2)] += random.randint(10, 20)  # add density
        fluid.veloc[int(N / 2 - 2), int(N / 2 - 2)] += [np.cos(i * 0.2 * 0.5), np.sin(i * 0.2 * 0.5)]
        fluid.step()

        im.set_array(fluid.density)  # Set the image array from numpy array
        ax.set_title(f"map_size={N}, dt={dt}, iteration={it}, diffusion={diff}, viscosity={visc}\nFrame={i}",fontsize=9)

    # Initialize figure and subplot = 111
    fig = plt.figure()
    ax = fig.add_subplot()

    # Plot density
    im = plt.imshow(fluid.density, cmap="plasma", vmax=100, interpolation="bilinear")
    anim = animation.FuncAnimation(fig, update_im, interval=0)
    plt.show()


if __name__ == "__main__":
    main()
