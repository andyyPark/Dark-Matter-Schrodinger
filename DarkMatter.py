import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse 
import scipy.sparse.linalg
from scipy.integrate import trapz
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys

class Schrodinger(object):

    def __init__(self, **kwargs):
        self.set_coordinate(kwargs)
        self.set_constants(kwargs)
        self.psi0 = self.wavepacket(self.X, self.Y, self.xa, self.ya,
                               self.k0x, self.k0y)
        self.psi0 = self.psi0 / np.sqrt(self.normalization(self.psi0))
        self.V = self.set_potential(self.X, self.Y)


    def solve_alpha_beta_gamma(self):
        self.gamma = (self.density * (self.xf - self.x0) ** 2 * self.c ** 3 / 
        (self.mass * self.nDm * self.G ** 3)) ** (1./10)
        self.beta = self.c / (self.G * self.gamma ** 3)
        self.alpha = self.beta ** 2 * self.gamma
        print("gamma:", self.gamma)
        print("beta:", self.beta)
        print("alpha:", self.alpha)

    def compute_r(self):
        self.delta_r = np.sqrt(self.delta_x ** 2 + self.delta_y ** 2)
        #print("max delta r:", max(self.delta_r))
        #print("min delta r:", min(self.delta_r[:len(self.delta_r) - 1]))
        self.delta_r_avg = np.mean(self.delta_r[10 : len(self.delta_r) - 1])
        print("avg delta r:", self.delta_r_avg)

    def plot_r(self):
        file_format = ".png"
        plt.plot(self.t, self.delta_r)
        plt.title("r")
        plt.savefig("r" + str(self.c) + "-" + str(self.sigmax) + file_format, format='png')
        #plt.show()

    def compute_average_y(self, PSI):
        self.average_y = np.zeros(self.T)
        self.average_y2 = np.zeros(self.T)

        for n in range(0, self.T - 1):
            psi = PSI[:, n]
            self.average_y[n] = trapz(trapz(self.Y * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy)
            self.average_y2[n] = trapz(trapz(self.Y ** 2 * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy)

        self.delta_y = np.sqrt(self.average_y2 - self.average_y ** 2)
        #print("min delta y:", min(self.delta_y[: len(self.delta_y) - 1]))
        #print("max delta y:", max(self.delta_y))

    def plot_average_y(self):
        fig, ax = plt.subplots(1, 3, figsize=(12, 7))
        ax[0].plot(self.t, self.average_y)
        ax[0].set_title('Average y')
        ax[0].set_yscale('log')
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$y$')
        ax[1].plot(self.t, self.average_y2)
        #ax[1].set_yscale('log')
        ax[1].set_title('Average $y^2$')
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$y^2$')
        ax[2].plot(self.t, self.delta_y)
        #ax[2].set_yscale('log')
        ax[2].set_title('Uncertainity $\Delta y$')
        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$\Delta y$')
        fig.savefig('Uncertainty_y', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
        plt.show()

    def compute_average_x(self, PSI):
        self.average_x = np.zeros(self.T)
        self.average_x2 = np.zeros(self.T)

        for n in range(0, self.T - 1):
            psi = PSI[:, n]
            self.average_x[n] = trapz(trapz(self.X * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy)
            self.average_x2[n] = trapz(trapz(self.X ** 2 * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy)

        self.delta_x = np.sqrt(self.average_x2 - self.average_x ** 2)
        #print("min delta x:", min(self.delta_x[: len(self.delta_x) - 1]))
        #print("max delta x:", max(self.delta_x))

    def plot_average_x(self):
        fig, ax = plt.subplots(1, 3, figsize=(12, 7))
        #print(max(average_x), min(average_x[:len(average_x) - 1]))
        ax[0].plot(self.t, self.average_x)
        ax[0].set_yscale('log')
        #ax[0].set_ylim([1, 2.7])
        ax[0].set_title('Average $x$')
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[1].plot(self.t, self.average_x2)
        #ax[1].set_yscale('log')
        ax[1].set_title('Average $x^2$')
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x^2$')
        ax[2].plot(self.t, self.delta_x)
        #ax[2].set_yscale('log')
        #ax[2].set_ylim([1, 2.6])
        ax[2].set_title('Uncertainity $\Delta x$')
        ax[2].set_xlabel('$t$')
        ax[2].set_ylabel('$\Delta x$')
        fig.savefig('Uncertainty_x', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
        plt.show()

    def plot_normalization(self, PSI):
        normalization = np.zeros(self.T)

        for n in range(0, self.T - 1):
            psi = PSI[:, n]
            normalization[n] = self.normalization(psi.reshape(self.J, self.L))
        plt.plot(self.t, normalization)
        plt.xlabel('$t$')
        plt.ylabel('$N$')
        #plt.yscale("log")
        #plt.ylim(top=1.3)
        plt.title('Normalization')
        plt.savefig('Normalization.png')
        plt.show()

    def play_video(self, PSI):

        def update(i, P, surf):
            ax.clear()
            psi = abs(P[:, i].reshape(self.J, self.L) ** 2)
            surf = ax.plot_surface(self.X, self.Y, psi, \
            rstride=1, cstride=1, cmap='plasma')
            ax.set_zlim(0, 0.3)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$|\Psi(x, y)|^2$')
            return surf
        
        frames = PSI.shape[1]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$|\Psi(x, y)|^2$')
        surf = ax.plot_surface(self.X, self.Y, abs(PSI[:, 0].reshape(self.J, self.L) ** 2), \
                rstride=1, cstride=1, cmap='plasma')
        ax.set_zlim(0, 0.3)
        ani = animation.FuncAnimation(fig, update, frames=frames,\
                fargs=(PSI, surf), \
                interval=30, blit=False)

        plt.show()

    def solve(self):
        U1, U2 = self.sparse_matrix()
        LU = scipy.sparse.linalg.splu(U1)
        PSI = np.zeros((self.N, 2000), dtype=complex)
        PSI[:, 0] = self.psi0.reshape(self.N)
        
        MAX_STEP = 2000
        step_index = 0
        count = 1
        delta = 60

        self.delta_r = list()

        while True:
            # if it reaches MAX steps
            if step_index == MAX_STEP - 1:
                print("Reached max step")
                break

            b = U2.dot(PSI[:, step_index])
            PSI[:, step_index + 1] = LU.solve(b)

            self.delta_r.append(np.sqrt(self.calculate_average_x(PSI[:, step_index]) ** 2 + \
                                                self.calculate_average_y(PSI[:, step_index]) ** 2))      

            # if it reaches bound (50)
            if np.abs(self.delta_r[step_index] - 50) <= 0.1:
                print("Reached bound")
                break

            # if the average of last 5 is within 0.1 of the average of the first 5
            if count >= delta:
                tol = np.abs((np.average(self.delta_r[count - delta : count - int(delta/2)]) \
                     - np.average(self.delta_r[count - int(delta / 2) : count])) \
                          / np.average(self.delta_r[count - delta : count - int(delta / 2)])) 
                if tol <= 0.01:
                    print("Avg no change")
                    break

            step_index = step_index + 1
            count = count + 1


        '''
        for n in range(0, self.T - 1):
            b = U2.dot(PSI[:, n])
            PSI[:, n + 1] = LU.solve(b)

            
            old code

        '''

        filename = str(self.nDm) + "particles.txt"

        with open(filename, "a") as txtfile:
            if step_index != MAX_STEP - 1:
                txtfile.write("C-" + str(self.c) + ", Init Spread-" + str(self.sigma) + ": "\
                    + str(round(self.delta_r[-1], 4)) \
                    + " with " + str(count) + " steps\n")

                image_format = ".png"
                plt.plot(np.array(range(count)), self.delta_r)
                plt.title("r")
                plt.savefig("r" + str(self.nDm) + "-" + str(self.c) + "-" + str(self.sigmax) + image_format, format='png')
            else:
                txtfile.write("C-" + str(self.c) + ", Init Spread-" + str(self.sigma) + ": MAX\n")

        return PSI

    def calculate_average_x(self, psi):
        return np.sqrt(trapz(trapz(self.X ** 2 * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy)
        - trapz(trapz(self.X * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy) ** 2)

    def calculate_average_y(self, psi):
        return np.sqrt(trapz(trapz(self.Y ** 2 * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy)
        - trapz(trapz(self.Y * abs(psi.reshape(self.J, self.L) ** 2), self.x, dx=self.x), self.y, dx=self.dy) ** 2)

    def sparse_matrix(self):
        b = 1 + 1j * self.dt * self.hbar ** 2 * (1 / (self.dx ** 2.0) + 1 / (self.dy ** 2.0)) \
            + 1j * self.dt * self.V.reshape(self.N) / (2 * self.hbar)
        c = -1j * self.dt * self.hbar / (4 * self.mass * self.dx ** 2) * np.ones(self.N, dtype=complex)
        a = c
        d = -1j * self.dt * self.hbar / (4 * self.mass * self.dy ** 2) * np.ones(self.N, dtype=complex)
        e = d

        f = 1 - 1j * self.dt * self.hbar ** 2 * (1 / (self.dx ** 2) + 1 / (self.dy ** 2)) \
            - 1j * self.dt * self.V.reshape(self.N) / (2 * self.hbar)
        g = 1j * self.dt * self.hbar / (4 * self.mass * self.dx ** 2) * np.ones(self.N, dtype=complex)
        h = g
        k = 1j * self.dt * self.hbar / (4 * self.mass * self.dy ** 2) * np.ones(self.N, dtype=complex)
        p = k

        U1 = np.array([c, e, b, d, a])
        diags = np.array([-self.J, -1, 0, 1, self.J])
        A = scipy.sparse.spdiags(U1, diags, self.N, self.N)
        A = A.tocsc()

        U2 = np.array([h, p, f, k, g])
        B = scipy.sparse.spdiags(U2, diags, self.N, self.N)
        B = B.tocsc()

        return (A, B)
        
    def set_potential(self, x, y):
        '''
        dm = np.array([99.5, 99.5])
        r = np.sqrt((x - dm[0]) ** 2 + (y - dm[1]) ** 2)
        potential = self.G * self.mass / r
        '''
    
        n = int(np.sqrt(self.nDm))
        m = int(self.nx / n)
        dmList = np.array( [ (i * m - 0.5, j * m - 0.5) for i in range(-n, n + 1)  for j in range(-n, n + 1)])
        #print(dmList)
        potential = np.zeros((self.J, self.L))
        for dm in dmList:
            r = np.sqrt((x - dm[0]) ** 2 + (y - dm[1]) ** 2)
            potential = potential +  self.mass / r

        return -self.c * potential
        

    def normalization(self, psi):
        return np.abs(trapz(trapz(abs(psi ** 2), self.x, dx=self.dx), self.y, dx=self.dy))

    def wavepacket(self, x, y, xa, ya, k0x, k0y):
        N = 1.0 / (2 * np.pi * self.sigmax * self.sigmay)
        e1x = np.exp(-(x - xa) ** 2.0 / (2.0 * self.sigmax ** 2.0))
        e1y = np.exp(-(y - ya) ** 2.0 / (2.0 * self.sigmay ** 2.0))
        e2 = np.exp(1j * k0x * x + 1j * k0y * y)
        return N * e1x * e1y * e2

    def set_constants(self, args):
        self.mass = args['mass']
        self.hbar = args['hbar']
        self.G = args['G']
        self.G = 6e-39 #GeV^-2
        self.density = 2e-42

        self.x0 = -200
        self.xf = 200
        self.y0 = -200
        self.yf = 200
        self.xa = 100
        self.ya = 100
        self.x = np.linspace(self.x0, self.xf, self.nx + 1)
        self.y = np.linspace(self.y0, self.yf, self.ny + 1)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.J = len(self.x)
        self.L = len(self.y)
        self.N = self.J * self.L

        try:
            self.nDm = int(sys.argv[1])
            self.c = float(sys.argv[2])
            self.sigma = float(sys.argv[3])
            self.sigmax = self.sigma
            self.sigmay = self.sigma

        except:
            print("Input not a number")
            sys.exit(1)

    def set_coordinate(self, args):
        self.nx = args['nx']
        self.ny = args['ny']
        self.t0 = args['t0']
        self.tf = args['tf']
        self.dt = args['dt']
        self.k0x = args['k0x']
        self.k0y = args['k0y']
        self.Nt = int(round(self.tf / float(self.dt)))
        self.t = np.linspace(self.t0, self.Nt * self.dt, self.Nt)
        self.T = len(self.t)
        self.debug = False

if __name__ == '__main__':
    args = {
        'nx': 200,
        'ny': 200, 
        'x0': 0, 
        'xf': 40, 
        'y0': 0,
        'yf': 40,
        'xa': 10.0, 
        'ya': 10.0,
        't0': 0, 
        'tf': 48.0,
        'dt': 0.04, 
        'sigmax': 1e-35, 
        'sigmay': 1e-35,
        'k0x': 0.0,
        'k0y': 0.0,
        'mass': 0.5,
        'hbar': 1.0,
        'G': 1.0
        }
    schrodinger = Schrodinger(**args)
    PSI = schrodinger.solve()
    #schrodinger.play_video(PSI)
    #schrodinger.plot_normalization(PSI)
    #schrodinger.compute_average_x(PSI)
    #schrodinger.compute_average_y(PSI)
    #schrodinger.plot_average_x()
    #schrodinger.plot_average_y()
    #schrodinger.compute_r()
    #schrodinger.plot_r()
    #schrodinger.solve_alpha_beta_gamma()