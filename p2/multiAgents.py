# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        eval = successorGameState.getScore()
        while newScaredTimes:
            time = newScaredTimes.pop()
            ghost_state = newGhostStates.pop()
            dis = util.manhattanDistance(newPos, ghost_state.getPosition())
            if time == 0:
                if dis < 2:
                    eval = 0
            else:
                if dis <= time:
                    eval += 1 / dis * 10
        food = newFood.asList().copy()
        temp = 100
        while food:
            temp = min(temp, util.manhattanDistance(food.pop(), newPos))
        eval += 1 / temp

        return eval


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class Node:
    def __init__(self, state, agentIndex, level, Max):
        self.agentIndex = agentIndex
        self.state = state
        self.level = level
        self.Max = Max


class Move:
    def __init__(self, moveAction, moveScore):
        self.moveAction = moveAction
        self.moveScore = moveScore


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def nodeGetValue(self, node):
        numAgents = node.state.getNumAgents()
        depth = self.depth
        setLevel = numAgents * depth
        if node.Max:
            init_score = -9999
        else:
            init_score = 9999
        if node.state.isWin() or node.state.isLose() \
                or node.level == setLevel:
            return self.evaluationFunction(node.state)
        else:
            best = Move(None, init_score)
            for action in node.state.getLegalActions(node.agentIndex):
                temp_state = \
                    node.state.generateSuccessor(node.agentIndex, action)
                agentIndex = (node.agentIndex + 1) % numAgents
                newNode = Node(temp_state, agentIndex, node.level + 1,
                               agentIndex == 0)
                move = Move(action, None)
                move.moveScore = self.nodeGetValue(newNode)
                if node.Max:
                    if move.moveScore > best.moveScore:
                        best = move
                else:
                    if move.moveScore < best.moveScore:
                        best = move

            return best.moveScore

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        best = Move(None, -9999)
        init_actions = gameState.getLegalActions(0)
        for action in init_actions:
            temp_state = gameState.generateSuccessor(0, action)
            node = Node(temp_state, 1, 1, False)
            score = self.nodeGetValue(node)
            move = Move(action, score)
            if move.moveScore > best.moveScore:
                best = move

        return best.moveAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -9999
        beta = 9999
        node = Node(gameState, 0, 0, True)
        best = self.alpha_beta_node_get_value(node, alpha, beta)

        return best.moveAction

    def alpha_beta_node_get_value(self, node, alpha, beta):
        numAgents = node.state.getNumAgents()
        depth = self.depth
        setLevel = numAgents * depth
        if node.Max:
            init_score = -9999
        else:
            init_score = 9999
        if node.state.isWin() or node.state.isLose() \
                or node.level == setLevel:
            return Move(None, self.evaluationFunction(node.state))
        else:
            best = Move(None, init_score)
            for action in node.state.getLegalActions(node.agentIndex):
                temp_state = \
                    node.state.generateSuccessor(node.agentIndex, action)
                agentIndex = (node.agentIndex + 1) % numAgents
                newNode = Node(temp_state, agentIndex, node.level + 1,
                               agentIndex == 0)
                move = Move(action, None)
                move.moveScore = self.abnodeGetValue(newNode, alpha, beta).moveScore
                if node.Max:
                    if move.moveScore > best.moveScore:
                        best = move
                    if best.moveScore > beta:
                        return best
                    alpha = max(alpha, best.moveScore)
                else:
                    if move.moveScore < best.moveScore:
                        best = move
                    if best.moveScore < alpha:
                        return best
                    beta = min(beta, best.moveScore)

            return best


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        node = Node(gameState, 0, 0, True)
        best = self.expi_node_get_value(node)

        return best.moveAction
        util.raiseNotDefined()

    def expi_node_get_value(self, node):
        numAgents = node.state.getNumAgents()
        depth = self.depth
        setLevel = numAgents * depth
        if node.Max:
            init_score = -9999
        else:
            init_score = 0
        if node.state.isWin() or node.state.isLose() \
                or node.level == setLevel:
            return Move(None, self.evaluationFunction(node.state))
        else:
            best = Move(None, init_score)
            legalActions = node.state.getLegalActions(node.agentIndex)
            for action in legalActions:
                temp_state = \
                    node.state.generateSuccessor(node.agentIndex, action)
                agentIndex = (node.agentIndex + 1) % numAgents
                newNode = Node(temp_state, agentIndex, node.level + 1,
                               agentIndex == 0)
                move = Move(action, None)
                move.moveScore = self.expi_node_get_value(newNode).moveScore
                if node.Max:
                    if move.moveScore > best.moveScore:
                        best = move

                else:
                    best.moveScore += 1 / len(legalActions) * move.moveScore
            return best
        pass


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pacPos = currentGameState.getPacmanPosition()
    dis = 1000
    for foodPos in currentGameState.getFood().asList():
        dis = min(dis, util.manhattanDistance(pacPos, foodPos))

    return currentGameState.getScore() - dis / 3 - currentGameState.getNumFood() * 10
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
