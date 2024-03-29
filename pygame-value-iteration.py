#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import random
import math
import pygame

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
GRID_COLS = 5
GRID_ROWS = 5
START_STATE = (0,0)
GOAL_STATE = (4,4)
WATER_STATE = (4,2)
OBSTACLES = [(2,2),(3,2)]
TERMINAL_STATES = [WATER_STATE, GOAL_STATE]
STATES = [(i,j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
ACTIONS = ['U', 'D', 'L', 'R'] # up, down, left, right, no action

DISCOUNT_FACTOR = 0.9
THRESHOLD = 0.0001
MAX_EPISODES = 10000

# pygame setup
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
BLOCK_SIZE = 100
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
WINDOW.fill((0,0,0))
pygame.display.set_caption("Gridworld")
ROBOT = pygame.image.load("robot.png").convert_alpha()
DRAW_STATES = {}
EPISODE_SPEED = 100 # milliseconds, must be integer


# ---- PYGAME FUNCTIONS ----

def drawGrid():
    margin = 1
    color = (255,255,255)
    states = [(i,j) for i in range(GRID_COLS) for j in range(GRID_ROWS)]

    i = 0
    for column in range(0, WINDOW_WIDTH+1, BLOCK_SIZE+margin):
        for row in range(0, WINDOW_HEIGHT+1, BLOCK_SIZE+margin):
            pygame.draw.rect(WINDOW, color, [row, column, BLOCK_SIZE, BLOCK_SIZE])
            DRAW_STATES[states[i]] = (row,column) # map state cord with pixel cord for drawing
            i += 1


# Takes list of obstacle cords and draws those cells red
def drawObstacles(obstacles_list):
    color = (255,0,0) # red

    # draw red obstacles
    for obstacleCord in obstacles_list:
        x = DRAW_STATES[obstacleCord][0]
        y = DRAW_STATES[obstacleCord][1]
        pygame.draw.rect(WINDOW, color, [x, y, BLOCK_SIZE, BLOCK_SIZE])

    # draw blue water state
    pygame.draw.rect(WINDOW, (30,144,255), [DRAW_STATES[WATER_STATE][0], DRAW_STATES[WATER_STATE][1], BLOCK_SIZE, BLOCK_SIZE])


# Draws the goal cord green
def drawGoal(goalCord):
    color = (124,252,0) # green
    x = DRAW_STATES[goalCord][0]
    y = DRAW_STATES[goalCord][1]
    pygame.draw.rect(WINDOW, color, [x, y, BLOCK_SIZE, BLOCK_SIZE])


# Draws agent on the window at cord
def drawAgent(cord):
    x = DRAW_STATES[cord][0]
    y = DRAW_STATES[cord][1]
    WINDOW.blit(ROBOT, ROBOT.get_rect(center=(x+50, y+50)))


# Displays iteration text
def displayIteration(i, font):
    counter_text = font.render("Iteration: " + str(i), True, (0, 0, 0))
    WINDOW.blit(counter_text, (5, 5))


# --- GRIDWORLD FUNCTIONS ---

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
    policy = {}

    iteration = 0
    converged = False

    while not converged:
        delta = 0
        newV = rewardFunction()

        for state in STATES:
            if state not in TERMINAL_STATES + OBSTACLES:
                actionValues = [calculateValue(V, state, action) for action in ACTIONS]
                newV[state] = round(max(actionValues), 4)
                bestAction = ACTIONS[np.argmax(actionValues)]
                policy[state] = bestAction
                delta = max(delta, abs(newV[state] - V[state]))

        V = newV
        
        print(f"VI iteration {iteration}:")
        visualizeV(V)
        print()

        if delta < THRESHOLD:
            converged = True
            print(f"Converged after {iteration} iterations.")
            break

        iteration += 1

    return V, policy


def visualizeV(V):
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


def visualizePolicy(policy):
    bestActions = list(policy.values())

    # insert - for obstacles/terminal states
    bestActions.insert(12, "O")
    bestActions.insert(17, "O")
    bestActions.insert(22, "W")
    bestActions.insert(24, "G")

    i = 0
    row = []
    for bestAction in bestActions:
        row.append(bestAction)
        if i == GRID_SIZE-1:
            print(row)
            row = []
            i = 0
        else:
            i += 1


def runOptimalPolicy(policy):
    discountedReturns = []
    font = pygame.font.SysFont("Arial", 20)

    for episode in range(MAX_EPISODES):
        currState = START_STATE
        discountedReturn = 0
        timestep = 0

        while currState != GOAL_STATE:
            # allow exit from pygame window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # for display
            drawGrid()
            drawObstacles(OBSTACLES)
            drawGoal(GOAL_STATE)
            displayIteration(episode, font)

            # choose optimal action from policy
            action = policy[currState]

            # get next state and the reward for the action
            nextState, reward = takeAction(currState, action)

            drawAgent(nextState)
            pygame.display.update()
            pygame.time.wait(EPISODE_SPEED)

            # calculate discounted return
            discountedReturn += reward * (DISCOUNT_FACTOR ** timestep)
            timestep += 1

            # update current state to next state
            currState = nextState

            # if agent in water, go to next episode
            if currState == WATER_STATE:
                break

        # add the episodes final discounted return
        discountedReturns.append(discountedReturn)

    return discountedReturns


def main():
    pygame.init()

    V, policy = valueIteration()

    print()
    print("Optimal Value Function")
    visualizeV(V)

    print()
    print("Optimal Policy")   
    visualizePolicy(policy)

    returns = runOptimalPolicy(policy)

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    max_return = np.max(returns)
    min_return = np.min(returns)

    print()
    print(f"Mean discounted return: {mean_return:.2f}")
    print(f"Standard deviation of discounted returns: {std_return:.2f}")
    print(f"Maximum discounted return: {max_return:.2f}")
    print(f"Minimum discounted return: {min_return:.2f}")



if __name__ == '__main__':
    main()
