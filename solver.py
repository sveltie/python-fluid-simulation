import math
import numpy as np
# np.set_printoptions(threshold=np.inf)

class Fluid:
    def __init__(self, N, dt, it, diffusion, viscosity):
        self.size = N   # map size
        self.dt = dt    # time interval
        self.iter = it  # Iteration for solving linear equation
        self.diffusion = diffusion  # Diffusion
        self.viscosity = viscosity  # Viscosity

        self.s = np.full((self.size, self.size), 0, dtype=float)          # Previous density
        self.density = np.full((self.size, self.size), 0, dtype=float)    # Current density

        self.veloc = np.full((self.size, self.size, 2), 0, dtype=float)   # Current velocity
        self.veloc0 = np.full((self.size, self.size, 2), 0, dtype=float)  # Previous velocity

    def step(self):
        dt = self.dt
        visc = self.viscosity
        diff = self.diffusion
        s = self.s
        density = self.density
        veloc = self.veloc
        veloc0 = self.veloc0

        # Diffuse all two velocity components
        self.diffuse(veloc0, veloc, visc)

        # Fix up velocities so they keep things incompressible
        # Vx0, Vy0, Vx, Vy
        self.project(veloc0[:, :, 0], veloc0[:, :, 1], veloc[:, :, 0], veloc[:, :, 1])

        # Move the velocities around according to the velocities of the fluid
        self.advect(veloc[:, :, 0], veloc0[:, :, 0], veloc0)    # Vx, Vx0
        self.advect(veloc[:, :, 1], veloc0[:, :, 1], veloc0)    # Vy, Vxy

        # Fix up the velocities
        # Vx, Vy, Vx0, Vy0
        self.project(veloc[:, :, 0], veloc[:, :, 1], veloc0[:, :, 0], veloc0[:, :, 1])

        # Diffuse the density
        # size, density, diff
        self.diffuse(s, density, diff)

        # Move the density around according to the velocities
        self.advect(density, s, veloc)

    def set_bnd(self, vector):
        for i in range(1, self.size - 1):
            if len(vector.shape) > 2: # 3D vector
                vector[i, 0, 0] = vector[i, 1, 0]
                vector[i, 0, 1] = - vector[i, 1, 1]
                vector[i, -1, 0] = vector[i, -2, 0]
                vector[i, -1, 1] = - vector[i, -2, 1]

                # horizontal borders
                vector[0, i, 0] = - vector[1, i, 0]
                vector[0, i, 1] = vector[1, i, 1]
                vector[-1, i, 0] = - vector[-2, i, 0]
                vector[-1, i, 1] = vector[-2, i, 1]
            else: # 2D vector
                vector[i, 0] = vector[i, 1]
                vector[i, -1] = vector[i, -2]

                vector[0, i] = vector[1, i]
                vector[-1, i] = vector[-2, i]

        # Set the corners
        vector[0, 0] = 0.5 * (vector[1, 0] + vector[0, 1])
        vector[0, -1] = 0.5 * (vector[1, -1] + vector[0, -2])
        vector[-1, 0] = 0.5 * (vector[-2, 0] + vector[- 1, 1])
        vector[-1, -1] = 0.5 * (vector[-2, -1] + vector[-1, -2])

    def lin_solve(self, x, x0, a, c):
        cRecip = 1.0 / c
        for i in range(0, self.iter):
            for j in range(1, self.size - 1):
                for k in range(1, self.size - 1):
                    x[k, j] = (x0[k, j] + a * (x[k + 1, j] +
                                                       x[k - 1, j] + x[k, j + 1] + x[k, j - 1])) * cRecip

            self.set_bnd(x)

    def diffuse(self, x, x0, diff):
        if diff != 0:
            a = self.dt * diff * (self.size - 2) * (self.size - 2)
            self.lin_solve(x, x0, a, 1 + 6 * a)
        else:
            x[:, :] = x0[:, :]

    def project(self, velocX, velocY, p, div):
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                div[j, i] = -0.5 * (
                    velocX[j + 1, i] -
                    velocX[j - 1, i] +
                    velocY[j, i + 1] -
                    velocY[j, i - 1]
                ) / self.size
                p[j, i] = 0

        self.set_bnd(div)
        self.set_bnd(p)
        self.lin_solve(p, div, 1, 6)

        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                velocX[j, i] -= 0.5 * (p[j + 1, i] - p[j - 1, i]) * self.size
                velocY[j, i] -= 0.5 * (p[j, i + 1] - p[j, i - 1]) * self.size

        self.set_bnd(self.veloc)

    def advect(self, d, d0, velocity):
        dtx = self.dt * (self.size - 2)
        dty = self.dt * (self.size - 2)

        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):

                tmp1 = dtx * velocity[j, i, 0]
                tmp2 = dty * velocity[j, i, 1]
                x = j - tmp1
                y = i - tmp2

                if x < 0.5:
                    x = 0.5

                if x > self.size + 0.5:
                    x = self.size + 0.5

                j0 = math.floor(x)
                j1 = j0 + 1.0

                if y < 0.5:
                    y = 0.5

                if y > self.size + 0.5:
                    y = self.size + 0.5

                i0 = math.floor(y)
                i1 = i0 + 1.0

                s1 = x - j0
                s0 = 1.0 - s1
                t1 = y - i0
                t0 = 1.0 - t1

                j0j = int(j0)
                j1j = int(j1)
                i0j = int(i0)
                i1j = int(i1)

                d[j, i] = s0 * (t0 * d0[j0j, i0j] + t1 * d0[j0j, i1j]) + \
                    s1 * (t0 * d0[j1j, i0j] + t1 * d0[j1j, i1j])

        self.set_bnd(d)