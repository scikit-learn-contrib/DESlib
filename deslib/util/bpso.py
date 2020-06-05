# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import copy
from typing import List

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

    def __init__(self, position, inertia, c1, c2):
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
                 max_iter,
                 n_particles,
                 n_dim,
                 init_inertia,
                 final_inertia,
                 c1,
                 c2,
                 transfer_function='v-shaped',
                 max_iter_no_change=None,
                 random_state=None,
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
                self.n_iter_no_change_ = 0

    def optimize(self, fitness_function):
        """
        Run the PSO algorithm.

        Parameters
        ----------
        fitness_function : function
            Function used to estimate the fitness of a binary particle.

        Return
        ------
        gbest_ : Particle
            Global best solution from the whole swarm.
        """
        self._create_swarm()
        self.n_iter_no_change_ = 0
        self.iter_ = 0

        while not self._stop():
            # compute fitness of each particle
            for particle in self.particles_:
                particle.fitness = fitness_function(particle.position)

            self._update_gbest()
            self._update_pbest()
            self._update_velocity()
            self._self_update()
            self._update_binary_particles()
            self.iter_ = self.iter_ + 1
            self.n_iter_no_change_ += 1
        return self.gbest_

    def _stop(self):
        """
        Function to check if the optimization should stop.
        """
        # Early stopping
        if (self.max_iter_no_change is not None
                and self.n_iter_no_change_ >= self.max_iter_no_change):
            return True
        # Reached maximum number of iteration
        if self.iter_ >= self.max_iter:
            return True
