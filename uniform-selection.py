#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Abraham Couperus
# SWEN-711: Homework 4 - Question 1

import pygame
import sys
import numpy as np
import random
import math

# globals
GRID_ROWS = 5
GRID_COLS = 5
START_STATE = (0, 0)
GOAL_STATE = (4, 4)
OBSTACLES = [(2,2),(3,2)]
WATER_STATE = (4,2)
ALL_STATES = [(i,j) for i in range(GRID_COLS) for j in range(GRID_ROWS)]

DISCOUNT_FACTOR = 0.9
MAX_EPISODES = 10000

# Reward structure:
# 0 as default
# +10 for goal state
# -10 for water state
# -1 for hitting obstacle/attempt to leave grid

# Grid structure:
# (0,0) (0,1) (0,2) (0,3) (0,4)
# (1,0) (1,1) (1,2) (1,3) (1,4)
# (2,0) (2,1) (2,2) (2,3) (2,4)
# (3,0) (3,1) (3,2) (3,3) (3,4)
# (4,0) (4,1) (4,2) (4,3) (4,4)


def isValidState(stateCord):
    if (stateCord[0] >= 0) and (stateCord[0] <= (GRID_ROWS-1)):
        if (stateCord[1] >= 0) and (stateCord[1] <= (GRID_COLS-1)):
            if stateCord not in OBSTACLES:
                return True
    return False # obstacle or outside grid


def getActionSpace():
    actionSpace = ["up","left","right","down"]
    return actionSpace
        

def initRewardFunction():
    goalReward = 10
    waterReward = -10
    obstacleReward = -1
    defaultReward = 0
    rewards = {}

    for state in ALL_STATES:
        if state == GOAL_STATE:
            rewards[state] = goalReward
        elif state == WATER_STATE:
            rewards[state] = waterReward
        elif state in OBSTACLES:
            rewards[state] = obstacleReward
        else:
            rewards[state] = defaultReward

    return rewards


# move agent up, down, left, or right given current state cord
# returns next state cord, and reward from the action taken
def makeAction(currStateCord, action):
    reward = 0 # default

    if action == "up":
        nextStateCord = (currStateCord[0]-1, currStateCord[1])
    elif action == "down":
        nextStateCord = (currStateCord[0]+1, currStateCord[1])
    elif action == "left":
        nextStateCord = (currStateCord[0], currStateCord[1]-1)
    elif action == "right":
        nextStateCord = (currStateCord[0], currStateCord[1]+1)
    else:
        return currStateCord, 0 # return 0 reward for invalid actions

    # make sure nextStateCord is within board boundaries AND not an obstacle
    if isValidState(nextStateCord):
        if nextStateCord == GOAL_STATE:
            reward = 10
            return nextStateCord, reward
        elif nextStateCord == WATER_STATE:
            reward = -10
            return nextStateCord, reward
        else:
            # give default
            return nextStateCord, reward
    else:
        # hit obstacle/attempt to leave grid
        reward = -1
        return currStateCord, reward


# Question 1 - Part 1 (Have the agent uniformly randomly select actions. Run 10,000 episodes.)
def uniformRandomSelection():
    discountedReturns = []

    for episode in range(MAX_EPISODES):
        currState = START_STATE
        discountedReturn = 0
        timestep = 0

        while currState != GOAL_STATE:

            # np.random uniformly distributes probablity between choices
            action = np.random.choice(getActionSpace())

            # get next state and the reward for the action
            nextState, reward = makeAction(currState, action)

            # calculate discounted return
            discountedReturn += reward * (DISCOUNT_FACTOR ** timestep)
            timestep += 1

            # update current state to next state
            currState = nextState

            # if agent gets stuck in water, go to next episode
            if currState == WATER_STATE:
                break

        # add the episodes final discounted return
        discountedReturns.append(discountedReturn)

    return discountedReturns


def main():
    returns = uniformRandomSelection()
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    max_return = np.max(returns)
    min_return = np.min(returns)

    print(f"Mean discounted return: {mean_return:.2f}")
    print(f"Standard deviation of discounted returns: {std_return:.2f}")
    print(f"Maximum discounted return: {max_return:.2f}")
    print(f"Minimum discounted return: {min_return:.2f}")


if __name__ == '__main__':
    main()