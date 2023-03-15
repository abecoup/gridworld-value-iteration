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
DRAW_STATES = {}

DISCOUNT_FACTOR = 0.9
MAX_EPISODES = 10
EPISODE_SPEED = 100 # milliseconds, must be integer

# pygame setup
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
BLOCK_SIZE = 100
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
WINDOW.fill((0,0,0))
pygame.display.set_caption("Gridworld")
ROBOT = pygame.image.load("robot.png").convert_alpha()


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


# ---- GRIDWORLD FUNCTIONS ----

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


def visualUniformRandomSelection():
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
            
            # np.random uniformly distributes probablity between choices
            action = np.random.choice(getActionSpace())

            # get next state and the reward for the action
            nextState, reward = makeAction(currState, action)

            # draw agent at next state, update display
            drawAgent(nextState)
            pygame.display.update()
            pygame.time.wait(EPISODE_SPEED)

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
    pygame.init()

    returns = visualUniformRandomSelection()
    
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