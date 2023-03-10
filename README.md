# Gridworld using Python/Pygame

This repository contains a Python implementation of a 5x5 grid-world environment using Pygame, where an agent (robot) navigates a grid world with obstacles and tries to reach the goal state. The environment dynamics take into account various probabilities for movement and confusion, as well as boundary and obstacle conditions.

The repository provides implementations of two algorithms: uniform random selection and value iteration. In the first algorithm, the agent uniformly randomly selects actions and runs for 10,000 episodes. In the second algorithm, the optimal policy is found using the value iteration algorithm, and the resulting policy is used to run the agent for 10,000 episodes. The mean, standard deviation, maximum, and minimum of the observed discounted returns are reported.
