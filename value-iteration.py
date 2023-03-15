#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import random
import math

# Value iteration implementation for Gridworld with following environment dynamics:
# 1. Probability of 0.8 the agent moves in a specified direction.
# 2. Probability of 0.05 it gets confused and veers to the right (i.e. -90deg from where it attempted to move)
# 3. Probability of 0.05 it gets confused and veers to the left (i.e. +90deg from where it attempted to move)
# 4. Probability of 0.10 the agent temporarily breaks and does not move at all
# 5. If dynamics would cause the agent to EXIT (leave grid boundary) or hit OBSTACLE then the agent does not move
# 6. Start in STATE = (0,0) and the process ends when STATE = (4,4)

# Grid structure:
# (0,0) (0,1) (0,2) (0,3) (0,4)
# (1,0) (1,1) (1,2) (1,3) (1,4)
# (2,0) (2,1) (2,2) (2,3) (2,4)
# (3,0) (3,1) (3,2) (3,3) (3,4)
# (4,0) (4,1) (4,2) (4,3) (4,4)

# Reward structure:
# 0 as default
# +10 for goal state
# -10 for water state
# -1 for hitting obstacle/attempt to leave grid

GRID_SIZE = 5
START_STATE = (0,0)
GOAL_STATE = (4,4)
WATER_STATE = (4,2)
OBSTACLES = [(2,2),(3,2)]
TERMINAL_STATES = [WATER_STATE, GOAL_STATE]
STATES = [(i,j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
ACTIONS = ['U', 'D', 'L', 'R'] # up, down, left, right, no action

DISCOUNT_FACTOR = 0.9
THRESHOLD = 0.0001


def inBounds(state):
    if (state[0] >= 0) and (state[0] <= (GRID_SIZE-1)):
        if (state[1] >= 0) and (state[1] <= (GRID_SIZE-1)):
            return True
    return False


def rewardFunction():
    rewards = {}
    goalReward = 10
    waterReward = -10
    obstacleReward = -1
    defaultReward = 0

    for state in STATES:
        if state == GOAL_STATE:
            rewards[state] = goalReward
        elif state == WATER_STATE:
            rewards[state] = waterReward
        elif state in OBSTACLES:
            rewards[state] = obstacleReward
        else:
            rewards[state] = defaultReward

    return rewards


def takeAction(state, action):
    R = rewardFunction()

    if action == "U":
        nextState = (state[0]-1, state[1])
    elif action == "D":
        nextState = (state[0]+1, state[1])
    elif action == "L":
        nextState = (state[0], state[1]-1)
    elif action == "R":
        nextState = (state[0], state[1]+1)
    elif action == "N": # stay still
        nextState = (state)

    if not inBounds(nextState) or nextState in OBSTACLES:
        return state, -1
    
    return nextState, R[nextState]


def goLeft(intendedAction):
    if intendedAction == 'U':
        return 'L'
    elif intendedAction == 'D':
        return 'R'
    elif intendedAction == 'R':
        return 'U'
    elif intendedAction == 'L':
        return 'D'

def goRight(intendedAction):
    if intendedAction == 'U':
        return 'R'
    elif intendedAction == 'D':
        return 'L'
    elif intendedAction == 'R':
        return 'D'
    elif intendedAction == 'L':
        return 'U'


def calculateValue(V, state, action):

    newStateIntended, reward = takeAction(state, action)
    newStateVeerLeft, reward = takeAction(state, goLeft(action))
    newStateVeerRight, reward = takeAction(state, goRight(action))
    stayState, reward = takeAction(state, "N")

    v = reward
    v += 0.8 * (DISCOUNT_FACTOR * V[newStateIntended]) # take intended action
    v += 0.05 * (DISCOUNT_FACTOR * V[newStateVeerLeft]) # veer left
    v += 0.05 * (DISCOUNT_FACTOR * V[newStateVeerRight]) # veer right
    v += 0.10 * (DISCOUNT_FACTOR * V[stayState]) # stay still

    return v



def valueIteration():
    V = rewardFunction() # initialize V to reward function

    iteration = 0
    converged = False

    while not converged:
        delta = 0
        newV = rewardFunction()

        for state in STATES:
            if state not in TERMINAL_STATES:
                newV[state] = round(max([calculateValue(V, state, action) for action in ACTIONS]), 4)
                delta = max(delta, abs(newV[state] - V[state]))

        V = newV
        
        print(f"VI iteration {iteration}:")
        visualize(V)
        print()

        if delta < THRESHOLD:
            converged = True
            print(f"Converged after {iteration} iterations.")
            break

        iteration += 1

    return V



def visualize(V):
    i = 0
    row = []
    for v in V.values():
        row.append(round(v, 2))
        if i == GRID_SIZE-1:
            print(row)
            row = []
            i = 0
        else:
            i += 1


def main():
    V = valueIteration()



if __name__ == '__main__':
    main()