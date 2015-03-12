# Gerald Brown (gemabrow@ucsc.edu) & Alfred Young (ayoung4@ucsc.edu) // CMPS 140 -- Winter 2015
# inference.py
# ------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import random
import capture
import game
from baselineTeam import ReflexCaptureAgent

class InferenceModule:
  """
  An inference module tracks a belief distribution over an enemy's location.
  """
  
  ############################################
  # Useful methods for all inference modules #
  ############################################
  
  def __init__(self, enemyIndex, myAgent):
    # a class representation of enemy agent that takes into
    # account both ghost and pacman states -- code at bottom
    self.enemyIsRed = True if enemyIndex in [0, 2] else False
    self.agent = myAgent
    self.enemy = EnemyAgent(enemyIndex, self.enemyIsRed)
    
  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the enemy from the given gameState.
    
    You must first place the enemy in the gameState, using setEnemyPosition below.
    """
    enemyPosition = gameState.getAgentPosition(self.enemy.index) # The position you set
    actionDist = self.enemy.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(enemyPosition, action)
      dist[successorPosition] = prob
    return dist
  
  def setEnemyPosition(self, gameState, enemyPosition):
    """
    Sets the position of the enemy for this inference module to the specified
    position in the supplied gameState.
    """
    conf = game.Configuration(enemyPosition, game.Directions.STOP)
    
    # If enemy's position falls in with their home side
    # enemy is a ghost, not a Pacman
    eX, eY = enemyPosition
    isEnemyPacman = False
    if self.enemyIsRed:
      isEnemyPacman = True if eX < self.halfWidth else False
    else:
      isEnemyPacman = True if eX > self.halfWidth else False
      
    print self.enemy.index, isEnemyPacman
    self.enemy.setPacman(isEnemyPacman)
    gameState.data.agentStates[self.enemy.index] = game.AgentState(conf, isEnemyPacman)
    return gameState
  
  def observeState(self, gameState):
    "Collects the relevant noisy distance observation and pass it along."
    distances = gameState.getAgentDistances()
    obs = distances[self.enemy.index]
    self.observe(obs, gameState)
      
  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    # given the layout and team color, returns a matrix of all positions
    # corresponding to that team color's side
    self.halfWidth = gameState.getWalls().width / 2
    self.initializeUniformly(gameState)

class ExactInference(InferenceModule):
  """
  The exact dynamic inference module should use forward-algorithm
  updates to compute the exact belief function at each time step.
  """
  
  def initializeUniformly(self, gameState):
    "Begin with a uniform distribution over enemy positions except for the initial enemypos"
    self.beliefs = util.Counter()
    for position in self.legalPositions: self.beliefs[position] = 1.0
    self.beliefs.normalize()
  
  def observe(self, observation, gameState):
    """
    Updates beliefs based on the distance observation and our agent's position.
    - The noisyDistance is the estimated manhattanDistance to the enemy you are tracking.
    - The emissionModel below calls the probability of the noisyDistance for any true 
      distance you supply. That is, it returns P(noisyDistance | TrueDistance)
    """
    noisyDistance = observation
    emissionModel = lambda tD: gameState.getDistanceProb(tD, noisyDistance)
    myAgentPos = gameState.getAgentPosition(self.agent.index)
    
    newBeliefs = util.Counter()
    # where p refers to legalPositions of an enemy
    for p in self.legalPositions:
      trueDistance = self.agent.distancer.getDistance(p, myAgentPos)
      if emissionModel(trueDistance) > 0:
        newBeliefs[p] = emissionModel(trueDistance) * self.beliefs[p]

    newBeliefs.normalize()
    self.beliefs = newBeliefs
    
  def elapseTime(self, gameState):
    """
    Update self.beliefs in response to a time step passing from the current state.
    """
    newBeliefs = util.Counter()
    
    for p in self.legalPositions:
      updatedState = self.setEnemyPosition(gameState, p)
      newPosDist = self.getPositionDistribution(updatedState)
      for newPos, prob in newPosDist.items():
        newBeliefs[newPos] += prob * self.beliefs[p]
        
    self.beliefs = newBeliefs

  def getBeliefDistribution(self):
    return self.beliefs

class EnemyAgent(ReflexCaptureAgent):
  """
  An agent representation of the enemy that takes on different personas
  given whether it is in a ghost or pacman state
  """
  def __init__( self, index, red, prob = 0.8):
    self.index = index
    self.red = red
    self.redFactor = 1 if red else -1
    self.isPacman = False
    self.bestProb = prob
    
  def setPacman( self, isPacman ):
    self.isPacman = isPacman
    
  def getBestActions(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return bestActions

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    if self.isPacman:
      features['successorScore'] = successor.getScore() * self.redFactor

      # Compute distance to the nearest food
      foodList = self.getFood(successor).asList()
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
        myPos = successor.getAgentState(self.index).getPosition()
        minDistance = min([util.manhattanDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = minDistance 
    else:
      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()

      # Computes whether we're on defense (1) or offense (0)
      features['onDefense'] = 1
      if myState.isPacman: features['onDefense'] = 0

      # Computes distance to invaders we can see
      enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
      features['numInvaders'] = len(invaders)
      if len(invaders) > 0:
        dists = [self.manhattanDistance(myPos, a.getPosition()) for a in invaders]
        features['invaderDistance'] = min(dists)

      if action == game.Directions.STOP: features['stop'] = 1
      rev = game.Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1
    return features

  def getWeights(self, gameState, action):
    if self.isPacman:
      return {'successorScore': 100, 'distanceToFood': -1}
    else:
      return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  def getDistribution( self, gameState ):
    legalActions = [action for action in gameState.getLegalActions( self.index ) if not game.Directions.STOP]
    bestActions = self.getBestActions(gameState)
    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = self.bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-self.bestProb ) / len(legalActions)
    dist.normalize()
    return dist