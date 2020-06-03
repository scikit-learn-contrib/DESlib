# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import copy
from typing import List
from typing import Optional
from typing import Union

import numpy as np

# Limits
X_MAX = 10
X_MIN = -X_MAX
MI = 100
POS_MAX = 100
POS_MIN = -100

# Auxiliary variables
z = 0


def s_shaped_transfer(X):
    result = 1.0 / (1.0 + np.power(np.e, -2.0 * X))
    result[np.isnan(result)] = 1
    return result


def v_shaped_transfer(X):
    return np.abs((2.0 / np.pi) * np.arctan((np.pi / 2.0) * X))


class Particle:
    """
    Class representing a particle in a swarm.

    Parameters
    ----------
    inertia : float
        Initial inertia of the swarm

    c1 : float
        Self coefficient

    c2 : float
        Group coefficient

    Attributes
    ----------
    n_dimensions : int
        Particle dimensionality
    pbest : array-like
        Particle best position
    best_fitness : float
        Best fitness values obtained by the particle
    fitness : float
        Current fitness value from the particle
    velocity :
        Velocity vector. Each element corresponds to the velocity in the
        corresponding dimension.
    phi : float
        Coefficient
    history : List[Float]
        Fitness evolution of the given particle.
    """

    def __init__(self,
                 position: Union[List[float], np.ndarray],
                 inertia: float,
                 c1: float,
                 c2: float,
                 ):
        self.position = np.asarray(position)
        self.c1 = c1
        self.c2 = c2
        self.inertia = inertia

        # class variables
        self.n_dimensions = position.size
        self.best_fitness = None
        self.fitness = None
        self.phi = 0
        self.pbest = np.copy(self.position)
        self.velocity = np.zeros(self.n_dimensions)
        self.history = []


class BPSO:
    """
    Bibary Particle Swarm Optimization (BPSO) with self updating mechanism.
    Conversion from continuous to binary representation is conducted using
    either the V-shaped and S-shaped transfer functions

    Parameters
    ----------
    max_iter : int, default 100
        Number of iterations in the optimization.
    n_particles : int, default 20
        Number of particles used in the optimization.
    init_inertia : float
        Initial inertia of the swarm
    final_inertia : float
        Final inertia of the swarm
    c1 : float
        Self coefficient
    c2 : float
        Group coefficient

    Attributes
    ----------
    n_particles_ : int
        Number of particles in the swarm
    particles_ : List[Particle]
        List of particles in the swarm.
    g_best_ : Particle
        Particle containing the best fitness in the swarm history

    References
    ----------
    Kennedy, James, and Russell Eberhart. "Particle swarm optimization."
    In Proceedings of IJCNN'95-International Conference on Neural Networks,
    vol. 4, pp. 1942-1948. IEEE, 1995.

    Mirjalili, Seyedali, and Andrew Lewis. "S-shaped versus V-shaped transfer
    functions for binary particle swarm optimization." Swarm and Evolutionary
    Computation 9 (2013): 1-14.

    Zhang, Ying Chao, Xiong Xiong, and QiDong Zhang. "An improved self-adaptive
    PSO algorithm with detection function for multimodal function optimization
    problems." Mathematical Problems in Engineering 2013 (2013).
    """
    def __init__(self,
                 max_iter: int,
                 n_particles: int,
                 n_dim: int,
                 init_inertia: float,
                 final_inertia: float,
                 c1: float,
                 c2: float,
                 transfer_function: str = 'v-shaped',
                 max_iter_no_change=None,
                 random_state: Optional[int] = None,
                 ):
        self.max_iter = max_iter
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.init_inertia = init_inertia
        self.final_inertia = final_inertia
        self.initial_c1 = c1
        self.initial_c2 = c2
        self.transfer_function = transfer_function
        self.verbose = verbose
        self.max_iter_no_change = max_iter_no_change
        self.random_state = random_state

    def _create_swarm(self):

        self.particles_ = []
        self.gbest_ = None

        positions = np.random.uniform(0, 1, (self.n_particles, self.n_dim))
        positions = (positions > 0.5).astype(int)
        for idx in range(self.n_particles):
            particle = Particle(positions[idx],
                                inertia=self.init_inertia,
                                c1=self.initial_c1,
                                c2=self.initial_c2)

            self.particles_.append(particle)

    def _update_velocity(self):
        """
        Update the velocity of each particle.
        """
        for particle in self.particles_:
            for dim in range(len(particle.position)):
                tmp_c1 = particle.pbest[dim] - particle.position[dim]
                tmp_c2 = self.gbest_.position[dim] - particle.position[dim]

                inertia = particle.inertia * particle.velocity[dim]
                cognitive = (
                        (particle.c1 * np.random.rand()) * tmp_c1)
                social = (particle.c2 * np.random.rand()) * tmp_c2

                particle.velocity[dim] = inertia + cognitive + social

                # Limit velocity
                if particle.velocity[dim] >= X_MAX:
                    particle.velocity[dim] = X_MAX
                elif particle.velocity[dim] <= X_MIN:
                    particle.velocity[dim] = X_MIN

    def _update_particles(self):

        for particle in self.particles_:
            for dim in range(len(particle.position)):
                particle.position[dim] = particle.position[dim] + \
                                         particle.velocity[dim]
                if particle.position[dim] >= POS_MAX:
                    particle.position[dim] = POS_MAX
                elif particle.position[dim] <= POS_MIN:
                    particle.position[dim] = POS_MIN

    def _update_binary_particles(self):
        for particle in self.particles_:
            velocity = self._transfer_function(particle.velocity)
            pos = (np.random.rand(self.n_dim) < velocity).astype(np.int)
            particle.position[pos == 1] = particle.position[pos == 1] ^ 1

    def _transfer_function(self, velocity):
        if self.transfer_function == 's-shape':
            velocity = s_shaped_transfer(velocity)
        else:
            velocity = v_shaped_transfer(velocity)
        return velocity

    def _self_update(self):
        # Compute phi for each particle
        for particle in self.particles_:
            tmp1 = 0
            tmp2 = 0
            for j in range(len(particle.position)):
                tmp1 = tmp1 + self.gbest_.position[j] - particle.position[
                    j]
                tmp2 = tmp2 + particle.pbest[j] - particle.position[j]
                if tmp1 == 0:
                    tmp1 = 1
                if tmp2 == 0:
                    tmp2 = 1
            particle.phi = abs(tmp1 / tmp2)
            ln = np.log(particle.phi)
            tmp = particle.phi * (self.iter_ - ((1 + ln) * self.max_iter) / MI)
            particle.inertia = ((self.init_inertia - self.final_inertia) / (
                    1 + np.exp(tmp))) + self.final_inertia
            particle.c1 = self.initial_c1 * (particle.phi ** (-1))
            particle.c2 = self.initial_c2 * particle.phi

    def _update_pbest(self):
        """
        Method used to update the position of each particle.
        """
        for particle in self.particles_:
            if (particle.best_fitness is None or
                    particle.best_fitness >= particle.fitness):
                particle.pbest = particle.position
                particle.best_fitness = particle.fitness

    def _update_gbest(self):
        """
        Method used to update the best particle in the swarm.
        """
        for particle in self.particles_:
            if self.gbest_ is None or particle.fitness < self.gbest_.fitness:
                self.gbest_ = copy.deepcopy(particle)
                self._n_iter_no_change = 0

    def optimize(self):
        """
        Run the PSO algorithm.

        Return
        ------
        gbest_ : Particle
            Particle with the best fitness value.

        """
        self._create_swarm()
        self._n_iter_no_change = 0
        self.iter_ = 0

        while not self._stop():
            self.iter_ = self.iter_ + 1
            self._n_iter_no_change += 1
            self._compute_fitness()
            self._update_gbest()
            self._update_pbest()
            self._update_velocity()
            self._self_update()
            self._update_binary_particles()

        return self.gbest_

    def _stop(self):
        """
        Function to check if the optimization should stop.
        """
        # check early stopping
        if (self.max_iter_no_change is not None
                and self._n_iter_no_change >= self.max_iter_no_change):
            return True
        # check reached maximum number of iteration
        if self.iter_ >= self.max_iter:
            return True

    @staticmethod
    def fitness_function(position):
        """
        Compute fitness

        Parameters
        ----------
        position : Numpy array
            A particle in the swarm

        Returns
        -------
        fitness : float
            Fitness of the particle.

        """
        return np.sum(position == 1)

    def _compute_fitness(self):
        """
        Compute the fitness of each particle
        """
        for particle in self.particles_:
            particle.fitness = self.fitness_function(
                particle.position)

    @staticmethod
    def fitness(particle, X, y, metric='euclidean', gamma=0.5):
        """X must be normalized a priori"""
        X_p = X[:, particle]
        score = BPSO.compute_knn_score(X_p, y, metric)
        distance = BPSO.computer_inner_outer_distances(X_p, y, metric)
        fitness = ((gamma * score) + ((1 - gamma) * distance))
        return fitness


def main():
    swarm = BPSO(1000, 10, 200, 1, 0.3, c1=2, c2=2, max_iter_no_change=50,)
    swarm.optimize()