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
import captureAgents
import game

class InferenceModule:
  """
  An inference module tracks a belief distribution over an enemy's location.
  """
  
  ############################################
  # Useful methods for all inference modules #
  ############################################
  
  def __init__(self, enemyIndex, myAgent):
    "Sets the enemy agent and team indices for later access"
    self.enemy = enemyIndex
    self.centered = myAgent
    
  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the enemy from the given gameState.
    
    You must first place the enemy in the gameState, using setEnemyPosition below.
    """
    enemyPosition = gameState.getEnemyPosition(self.enemy) # The position you set
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
    
    isEnemyPacman = True
    # If enemy's position falls in with their home side
    # enemy is a ghost, not a Pacman
    eX, eY = enemyPosition
    if self.enemyGrid[eX][eY]:
      isEnemyPacman = False
    gameState.data.agentStates[self.enemy] = game.AgentState(conf, isEnemyPacman)
    return gameState
  
  def observeState(self, gameState):
    "Collects the relevant noisy distance observation and pass it along."
    distances = gameState.getAgentDistances()
    
    # ------------------check logic on this part ------------------------------------
    #if len(distances) >= self.enemy: # Check for missing observations
    #  obs = distances[self.index - 1]
    obs = distances[self.enemy]
    self.observe(obs, gameState)
      
  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.enemyIsRed = gameState.isOnRedTeam(self.enemy)
    # given the layout and team color, returns a matrix of all positions
    # corresponding to that team color's side
    self.enemyGrid = gameState.halfGrid(gameState.getWalls(), self.enemyIsRed)
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
    intialEnemyPos = gameState.getInitialAgentPosition(self.enemy)
    self.beliefs[initialEnemyPos] += 1
    self.beliefs.normalize()
  
  def observe(self, observation, gameState):
    """
    Updates beliefs based on the distance observation and our agent's position.
    - The noisyDistance is the estimated manhattanDistance to the enemy you are tracking.
    - The emissionModel below calls the probability of the noisyDistance for any true 
      distance you supply. That is, it returns P(noisyDistance | TrueDistance)
    """
    noisyDistance = observation
    emissionModel = lambda tD: return gameState.getDistanceProb(tD, noisyDistance)
    myAgentPos = self.centered.getPosition()
    
    newBeliefs = util.Counter()
    # where p refers to legalPositions of an enemy
    for p in self.legalPositions:
      trueDistance = self.centered.getMazeDistance(p, myAgentPos)
      if emissionModel(trueDistance) > 0:
        newBeliefs[p] = emissionModel(trueDistance) * self.beliefs[p]

    newBeliefs.normalize()
    self.beliefs = newBeliefs
    
  def elapseTime(self, gameState):
    """
    Update self.beliefs in response to a time step passing from the current state.
    
    The transition model is not entirely stationary: it may depend on Pacman's
    current position (e.g., for DirectionalGhost).  However, this is not a problem,
    as Pacman's current position is known.

    In order to obtain the distribution over new positions for the
    enemy, given its previous position (oldPos) as well as Pacman's
    current position, use this line of code:

      newPosDist = self.getPositionDistribution(self.setEnemyPosition(gameState, oldPos))

    Note that you may need to replace "oldPos" with the correct name
    of the variable that you have used to refer to the previous enemy
    position for which you are computing this distribution.

    newPosDist is a util.Counter object, where for each position p in self.legalPositions,
    
    newPostDist[p] = Pr( enemy is at position p at time t + 1 | enemy is at position oldPos at time t )

    (and also given Pacman's current position).  You may also find it useful to loop over key, value pairs
    in newPosDist, like:

      for newPos, prob in newPosDist:
        ...

    As an implementation detail (with which you need not concern
    yourself), the line of code above for obtaining newPosDist makes
    use of two helper methods provided in InferenceModule above:

      1) self.setEnemyPosition(gameState, enemyPosition)
          This method alters the gameState by placing the enemy we're tracking
          in a particular position.  This altered gameState can be used to query
          what the enemy would do in this position.
      
      2) self.getPositionDistribution(gameState)
          This method uses the enemy agent to determine what positions the enemy
          will move to from the provided gameState.  The enemy must be placed
          in the gameState with a call to self.setEnemyPosition above.
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

