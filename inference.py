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
    self.enemyIsRed = True if enemyIndex in (0, 2) else False
    self.agent = myAgent
    self.enemy = enemyIndex
      
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
    initialPos = gameState.getInitialAgentPosition(self.enemy)
    for position in self.legalPositions: self.beliefs[position] = 1.0
    self.beliefs.normalize()
  
  def observe(self, observation, gameState):
    """
    Updates beliefs based on the distance observation and our agent's position.
    - The noisyDistance is the estimated manhattanDistance to the enemy you are tracking.
    - The emissionModel below calls the probability of the noisyDistance for any true 
      distance you supply. That is, it returns P(noisyDistance | TrueDistance)
      where our observation is noisyDistance
    """
    noisyDistance = observation[self.enemy]
    newBeliefs = util.Counter()
    enemyPos = gameState.getAgentPosition(self.enemy)
    if enemyPos:
      newBeliefs[enemyPos] = 1.0
    else:
      emissionModel = lambda tD: gameState.getDistanceProb(tD, noisyDistance)
      myAgentPos = gameState.getAgentPosition(self.agent.index)
      # where p refers to legalPositions of an enemy
      for p in self.legalPositions:
        trueDistance = self.agent.distancer.getDistance(p, myAgentPos)
        # accounts for ghosts that have just been eaten
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
      if self.beliefs[p] > 0:
        prob = 1.0/len(self.legalPositions)
        newBeliefs[p] += prob * self.beliefs[p]
    newBeliefs.normalize()
    self.beliefs = newBeliefs
  
  def mostProbablePosition(self):
    return self.beliefs.argMax()
  
  def getBeliefDistribution(self):
    return self.beliefs
