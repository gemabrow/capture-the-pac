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

class InferenceModule:
  """
  An inference module tracks a belief distribution over an enemy's location.
  """
  
  ############################################
  # Useful methods for all inference modules #
  ############################################
  
  def __init__(self, enemy, myAgent):
    "Sets the enemy agent and team indices for later access"
    self.enemy = enemy
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
    if len(distances) >= self.enemy: # Check for missing observations
      obs = distances[self.index - 1]
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
    Updates beliefs based on the distance observation and Pacman's position.
    
    The noisyDistance is the estimated manhattan distance to the enemy you are tracking.
    
    The emissionModel below stores the probability of the noisyDistance for any true 
    distance you supply.  That is, it stores P(noisyDistance | TrueDistance).

    self.legalPositions is a list of the possible enemy positions (you
    should only consider positions that are in self.legalPositions).
    
    - After receiving a reading, the observe function is called, which must update 
    the belief at every position.
    - In the Pac-Man display, high posterior beliefs are represented by bright colors, 
    while low beliefs are represented by dim colors. You should start with a large 
    cloud of belief that shrinks over time as more evidence accumulates.
    - Beliefs are stored as util.Counter objects (like dictionaries) in a field called 
    self.beliefs, which you should update.
    - You should not need to store any evidence. The only thing you need to store in 
    ExactInference is self.beliefs.
    
    """
    noisyDistance = observation
    emissionModel = busters.getObservationDistribution(noisyDistance)
    pacmanPosition = gameState.getPacmanPosition()
    
    newBeliefs = util.Counter()
    # where p refers to legalPositions of a enemy
    for p in self.legalPositions:
      trueDistance = util.manhattanDistance(p, pacmanPosition)
      "*** YOUR CODE HERE ***"
      if emissionModel[trueDistance] > 0:
        # check logic for += vs = ********************************** [X]checked
        newBeliefs[p] = emissionModel[trueDistance] * self.beliefs[p]
    # B(X) is proportional to P(e|X) * B'(X)
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
          
    - A DirectionalGhost is easier to track because it is more predictable. After 
    running away from one for a while, your agent should have a good idea where it is.
    - We assume that enemys still move independently of one another, so while you can 
    develop all of your code for one enemy at a time, adding multiple enemys 
    should still work correctly.
    """
    newBeliefs = util.Counter()
    
    for p in self.legalPositions:
      updatedState = self.setEnemyPosition(gameState, p)
      newPosDist = self.getPositionDistribution(updatedState)
      for newPos, prob in newPosDist.items():
        newBeliefs[newPos] += prob * self.beliefs[p]
    
    # check logic for normalizing here *******************[X]checked
    # newBeliefs.normalize()
    self.beliefs = newBeliefs

  def getBeliefDistribution(self):
    return self.beliefs

class ParticleFilter(InferenceModule):
  """
  A particle filter for approximately tracking a single enemy.
  
  Useful helper functions will include random.choice, which chooses
  an element from a list uniformly at random, and util.sample, which
  samples a key from a Counter by treating its values as probabilities.
  
  Idea: Repeat a fixed Bayes net structure at each time
        Variables from time t can condition on those from t-1
  Procedure: "unroll" the network for T time steps, then eliminate
             variables until P(X_T|e_1:T) is computed
  Online belief updates: Eliminate all variables from the previous
                         time step; store factors for current time only
  """
  
  def initializeUniformly(self, gameState, numParticles=300):
    """
    Initializes a list of particles, with emplacing 
    an equal number of particle in each position
    """
    self.numParticles = numParticles
    "*** YOUR CODE HERE ***"
    self.particles = []
    # efficiency hack: since self.numParticles is already initialized,
    # use "numParticles" as a terminating loop value
    while numParticles > 0:
      for p in self.legalPositions:
        self.particles.append(p)
        numParticles -= 1
        if numParticles is 0:
          #print "break statement needed"
          break

    
  def observe(self, observation, gameState):
    """
    -- Weight each entire sample by the likelihood --
    --- of the evidence conditioned on the sample ---
    Update beliefs based on the given distance observation.
    Given: P(X_1), P(X_t+1 | X_t ), P(E_t | X_t)
    w(x) = P(e|x)
    B(X) proportional to P(e|X) * B'(X)
    1. Each sample is propagated forward by sampling the next state value
       given the current value for the sample, based on the transition model
    2. Each sample is weighted by the likelihood it assigns to the new evidence
    3. The population is resampled to generate a new population of N samples.
       Each new sample is selected from the current population; the probability
       that a particular sample is selected is proportional to its weight.
       The new samples are unweighted.
    """
    noisyDistance = observation
    emissionModel = busters.getObservationDistribution(observation)
    pacmanPosition = gameState.getPacmanPosition()
    #print noisyDistance
    "*** YOUR CODE HERE ***"
    newBeliefs = util.Counter()
    prevBeliefs = self.getBeliefDistribution() # get previous distribution of particles
    for p in self.legalPositions:
      trueDistance = util.manhattanDistance(p, pacmanPosition)
      newBeliefs[p] += emissionModel[trueDistance] * prevBeliefs[p]
    
    if newBeliefs.totalCount() == 0:
      # prevents list index out of range error
      "resample all particles from the prior --> uniform initial distribution"
      self.initializeUniformly(gameState, self.numParticles)
    else:
      # resample the population given the current weights
      updatedSample = []
      for i in range(self.numParticles):
        updatedSample.append( util.sample(newBeliefs) )
      self.particles = updatedSample
      
  def elapseTime(self, gameState):
    """
    -- Sample a successor for each particle --
    Update beliefs for a time step elapsing.

    As in the elapseTime method of ExactInference, you should use:

      newPosDist = self.getPositionDistribution(self.setEnemyPosition(gameState, oldPos))

    to obtain the distribution over new positions for the enemy, given
    its previous position (oldPos) as well as Pacman's current
    position.
    * newPosDist is a util.Counter object, where for each position p in self.legalPositions
    * newPostDist[p] = Pr( enemy is at position p at time t + 1 | enemy is at position oldPos at time t )
    x' = sample( P(X'|x) )
    Each particle is moved by sampling its next
    position from the transition model
    """
    "*** YOUR CODE HERE ***"
    updatedSample = []
    for particle in self.particles:
      updatedState = self.setEnemyPosition(gameState, particle)
      newPosDist = self.getPositionDistribution(updatedState)
      updatedSample.append( util.sample(newPosDist) ) 
    self.particles = updatedSample

  def getBeliefDistribution(self):
    """
    Return the agent's current belief state, a distribution over
    enemy locations conditioned on all evidence and time passage.
    """
    "*** YOUR CODE HERE ***"
    distribution = util.Counter()
    for particle in self.particles: distribution[particle] += 1.0
    distribution.normalize()
    return distribution

class MarginalInference(InferenceModule):
  "A wrapper around the JointInference module that returns marginal beliefs about enemys."

  def initializeUniformly(self, gameState):
    "Set the belief state to an initial, prior value."
    if self.index == 1: jointInference.initialize(gameState, self.legalPositions)
    jointInference.addGhostAgent(self.enemyAgent)
    
  def observeState(self, gameState):
    "Update beliefs based on the given distance observation and gameState."
    if self.index == 1: jointInference.observeState(gameState)
    
  def elapseTime(self, gameState):
    "Update beliefs for a time step elapsing from a gameState."
    if self.index == 1: jointInference.elapseTime(gameState)
    
  def getBeliefDistribution(self):
    "Returns the marginal belief over a particular enemy by summing out the others."
    jointDistribution = jointInference.getBeliefDistribution()
    dist = util.Counter()
    for t, prob in jointDistribution.items():
      dist[t[self.index - 1]] += prob
    return dist
  
class JointParticleFilter:
  "JointParticleFilter tracks a joint distribution over tuples of all enemy positions."
  
  def initialize(self, gameState, legalPositions, numParticles = 600):
    "Stores information about the game, then initializes particles."
    self.numGhosts = gameState.getNumAgents() - 1
    self.numParticles = numParticles
    self.enemyAgents = []
    self.legalPositions = legalPositions
    self.initializeParticles()
    
  def initializeParticles(self):
    "Initializes particles randomly.  Each particle is a tuple of enemy positions."
    self.particles = []
    for i in range(self.numParticles):
      self.particles.append(tuple([random.choice(self.legalPositions) for j in range(self.numGhosts)]))

  def addGhostAgent(self, agent):
    "Each enemy agent is registered separately and stored (in case they are different)."
    self.enemyAgents.append(agent)
    
  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the gameState.

    To loop over the enemys, use:

      for i in range(self.numGhosts):
        ...

    Then, assuming that "i" refers to the (0-based) index of the
    enemy, to obtain the distributions over new positions for that
    single enemy, given the list (prevGhostPositions) of previous
    positions of ALL of the enemys, use this line of code:

      newPosDist = getPositionDistributionForGhost(setEnemyPositions(gameState, prevGhostPositions),
                                                   i + 1, self.enemyAgents[i])

    Note that you may need to replace "prevGhostPositions" with the
    correct name of the variable that you have used to refer to the
    list of the previous positions of all of the enemys, and you may
    need to replace "i" with the variable you have used to refer to
    the index of the enemy for which you are computing the new
    position distribution.

    As an implementation detail (with which you need not concern
    yourself), the line of code above for obtaining newPosDist makes
    use of two helper functions defined below in this file:

      1) setEnemyPositions(gameState, enemyPositions)
          This method alters the gameState by placing the enemys in the supplied positions.
      
      2) getPositionDistributionForGhost(gameState, enemyIndex, agent)
          This method uses the supplied enemy agent to determine what positions 
          a enemy (enemyIndex) controlled by a particular agent (enemyAgent) 
          will move to in the supplied gameState.  All enemys
          must first be placed in the gameState using setEnemyPositions above.
          Remember: enemys start at index 1 (Pacman is agent 0).  
          
          The enemy agent you are meant to supply is self.enemyAgents[enemyIndex-1],
          but in this project all enemy agents are always the same.
    """
    newParticles = []
    for oldParticle in self.particles:
      newParticle = list(oldParticle) # A list of enemy positions
      "*** YOUR CODE HERE ***"
      # note that the length of list newParticle is
      # equal to the number of enemys agents
      for i, pos in enumerate(newParticle):
        updatedState = setEnemyPositions(gameState, newParticle)
        newPosDist = getPositionDistributionForGhost(gameState, i+1, self.enemyAgents[i])
        newParticle[i] = util.sample(newPosDist)
      newParticles.append(tuple(newParticle))
    self.particles = newParticles
  
  def observeState(self, gameState):
    """
    Resamples the set of particles using the likelihood of the noisy observations.

    As in elapseTime, to loop over the enemys, use:

      for i in range(self.numGhosts):
        ...

    A correct implementation will handle two special cases:
      1) When a enemy is captured by Pacman, all particles should be updated so
         that the enemy appears in its prison cell, position (2 * i + 1, 1),
         where "i" is the 0-based index of the enemy.

         You can check if a enemy has been captured by Pacman by
         checking if it has a noisyDistance of 999 (a noisy distance
         of 999 will be returned if, and only if, the enemy is
         captured).
         
      2) When all particles receive 0 weight, they should be recreated from the
          prior distribution by calling initializeParticles.
    """ 
    pacmanPosition = gameState.getPacmanPosition()
    noisyDistances = gameState.getNoisyGhostDistances()
    
    if len(noisyDistances) < self.numGhosts: return
    emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]

    "*** YOUR CODE HERE ***"
    import logging
    newBeliefs = util.Counter()
    for particle in self.particles:
      w = 1.0 # initial weight
      for j in range(self.numGhosts):
        if noisyDistances[j] == 999:
          # update particles to prison cell
          particle = self.getPrisonParticles(particle, j)
          # logging.warning('self.particles does not reflect captured enemys')
        else:
          trueDistance = util.manhattanDistance(particle[j], pacmanPosition)
          w *= emissionModels[j][trueDistance]
      newBeliefs[particle] += w
    
    newBeliefs.normalize()
    if newBeliefs.totalCount() == 0:
      self.initializeParticles()
    else:
      updatedSamples = []
      for i in range(self.numParticles):
          updatedSamples.append(util.sample(newBeliefs))
      self.particles = updatedSamples
      
  def getPrisonParticles(self, particles, enemyIndex):
    """
    Updates all associated particles with the ith enemy to
    its respective prison cell given position (2 * i + 1, 1)
    """
    p = list(particles)
    p[enemyIndex] = ( 2 * enemyIndex + 1, 1)
    return tuple(p)
    
  def getBeliefDistribution(self):
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    return dist

# One JointInference module is shared globally across instances of MarginalInference 
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, enemyIndex, agent):
  """
  Returns the distribution over positions for a enemy, using the supplied gameState.
  """
  enemyPosition = gameState.getEnemyPosition(enemyIndex) 
  actionDist = agent.getDistribution(gameState)
  dist = util.Counter()
  for action, prob in actionDist.items():
    successorPosition = game.Actions.getSuccessor(enemyPosition, action)
    dist[successorPosition] = prob
  return dist
  
def setEnemyPositions(gameState, enemyPositions):
  "Sets the position of all enemys to the values in enemyPositionTuple."
  for index, pos in enumerate(enemyPositions):
    conf = game.Configuration(pos, game.Directions.STOP)
    gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
  return gameState  

