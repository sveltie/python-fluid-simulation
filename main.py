"""
Based on:
Jos Stam Real-Time Fluid Dynamics       https://www.researchgate.net/publication/2560062_Real-Time_Fluid_Dynamics_for_Games       
Mikeash Fluid Simulation for Dummies    https://www.mikeash.com/pyblog/fluid-simulation-for-dummies.html
"""

import sys
import random
import numpy as np
from solver import Fluid


def main():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
        N = 64
        dt = 0.1
        it = 10 # iteration
        diff = 0.0000
        visc = 0.0000

        fluid = Fluid(N, dt, it, diff, visc)

        def update_im(i):
            fluid.density[int(N / 2 - 2):int(N / 2), int(N / 2 - 2):int(N / 2)] += random.randint(10, 20)  # add density
            fluid.veloc[int(N / 2 - 2), int(N / 2 - 2)] += [np.cos(i * 0.2 * 0.5), np.sin(i * 0.2 * 0.5)]
            fluid.step()
            im.set_array(fluid.density)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'N={N}, dt={dt}, iteration={it}, diffusion={diff}, viscosity={visc}', fontsize=9)
        im = plt.imshow(fluid.density, cmap='plasma', vmax=100, interpolation='bilinear')
        anim = animation.FuncAnimation(fig, update_im, interval=0)
        plt.show()
    
    except:
        plt.close()

if __name__ == "__main__":
    main()