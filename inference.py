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
import captureAgent
import game

class InferenceModule:
  """
  An inference module tracks a belief distribution over an enemy's location.
  """
  
  ############################################
  # Useful methods for all inference modules #
  ############################################
  
  def __init__(self, enemyAgent, myAgent):
    # a class representation of enemy agent that takes into
    # account both ghost and pacman states -- code at bottom
    self.enemy = enemyAgent
    self.centered = myAgent
    
  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the enemy from the given gameState.
    
    You must first place the enemy in the gameState, using setEnemyPosition below.
    """
    enemyPosition = gameState.getEnemyPosition(self.enemy.index) # The position you set
    actionDist = self.enemy.getDistribution(gameState, self.centered)
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
    isEnemyPacman = False if self.enemyGrid[eX][eY] else True
    self.enemy.setPacman(isEnemyPacman)
    gameState.data.agentStates[self.enemy.index] = game.AgentState(conf, isEnemyPacman)
    return gameState
  
  def observeState(self, gameState):
    "Collects the relevant noisy distance observation and pass it along."
    distances = gameState.getAgentDistances()
    
    # ------------------check logic on this part ------------------------------------
    #if len(distances) >= self.enemy: # Check for missing observations
    #  obs = distances[self.index - 1]
    obs = distances[self.enemy.index]
    self.observe(obs, gameState)
      
  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.enemyIsRed = gameState.isOnRedTeam(self.enemy.index)
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
    intialEnemyPos = gameState.getInitialAgentPosition(self.enemy.index)
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
      trueDistance = self.centered.distancer.getDistance(p, myAgentPos)
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

class EnemyAgent(Agent):
  """
  An agent representation of the enemy that takes on different personas
  given whether it is in a ghost or pacman state
  """
  def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8):
    self.index = index
    self.prob_attack = prob_attack
    self.prob_scaredFlee = prob_scaredFlee
    self.evaluationFunction = scoreEvaluation
    self.isPacman = False
    assert self.evaluationFunction != None
  
  def setPacman( self, isPacman ):
    self.isPacman = isPacman
  
  def getBestActions(state, myAgent):
    """
    Depending on my state and enemy state, return list of best actions
    """
    # Read variables from state (position is according to that SET by inferenceModule)
    agentState = state.getAgentState( self.index )
    legalActions = [for action in state.getLegalActions( self.index ) if not Directions.STOP]
    # Select best actions given the state
    bestActions = []
    if agentState.isPacman:
      successors = [(state.generateSuccessor( self.index, action), action) for action in legalActions]
      scored = [(self.evaluationFunction(state), action) for state, action in successors]
      bestScore = max(scored)[0]
      bestProb = self.prob_attack
      bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
    else:
      pos = agentState.getPosition()
      isScared = agentState.scaredTimer > 0
      speed = 0.5 if isScared else 1
      
      actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
      newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
      myPosition = state.getAgentPosition(myAgent.index)
      friendPosition = state.getAgentPosition(myAgent.friendIndex)
      
      distancesToUs = [MyAgent.distancer.getDistance( pos, myPosition ) for pos in newPositions]
      distancesToUs.append( MyAgent.distancer.getDistance( pos, friendPosition ) for pos in newPositions )
      if isScared:
        bestScore = max( distancesToUs )
        bestProb = self.prob_scaredFlee
      else:
        bestScore = min( distancesToUs )
        bestProb = self.prob_attack
      bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]
      
    return bestActions
      
  def getDistribution( self, state, myAgent):

    bestActions = getBestActions(state)
    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
    dist.normalize()
    return dist
  
def scoreEvaluation(state):
  return state.getScore()