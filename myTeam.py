# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'PrimaryAgent', second = 'SecondaryAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class BaseAgent(CaptureAgent):
  """
  A base agent to serve as a foundation for the varying agent structure.
  """
  
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 
    
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """
    CaptureAgent.registerInitialState(self, gameState)

    ''' 
    VARIABLES (from CaptureAgent):
    self.index = index for this agent
    self.red = true if you're on the red team, false otherwise
    self.agentsOnTeam = a list of agent objects that make up your team
    self.distancer = distance calculator (contest code provides this)
    self.observationHistory = list of GameState objects that correspond
        to the sequential order of states that have occurred so far this game
    self.timeForComputing = an amount of time to give each turn for computing maze distances
        (part of the provided distance calculator)
    '''
    
  '''
  FUNCTIONS (from CaptureAgent):
  self.final(gameState): 
    resets observationHistory
  self.registerTeam(agentsOnTeam): 
    fills CaptureAgent.agentsOnTeam with indices of agents on team
  self.observationFunction(gameState):
    return gameState.makeObservation(CaptureAgent.index)
  self.getAction(gameState):
    appends current gameState on to our observation history
    and calls our choose action method
  self.getFood(gameState):
    returns a matrix with the food we're meant to eat
    in the form m[x][y]==true if there is food for us
    in that square (based on our team color).
   self.getFoodYouAreDefending(gameState):
     returns the food we should protect in the form
     of a matrix m[x][y]==true if there is food our
     opponent can eat at those coordinates.
   self.getCapsules(gameState):
   self.getCapsulesYouAreDefending(gameState):
   self.getOpponents(gameState):
     returns agent indices of our opponents in list form.
   self.getTeam(gameState:
     returns a list of indices of the agents on our team.
   self.getScore(gameState):
     returns a number that is the difference in teams' scores.
     will be negative if we're a bunch of sissy la-la losers.
   self.getMazeDistance(pos1, pos2):
     returns the maze distance from pos1 to pos2.
   self.getPreviousObservation():
     returns the last GameState object this agent saw
     (may not include the exact locations of our opponent's agents)
   self.getCurrentObservation():
     like before, but now
   self.displayDistributionsOverPositions(distributions):
     arg distributions is a tuple or list of util.Counter objects,
     where the i'th Counter has keys that are board positions (x,y)
     and values that encode the probability that agent i is at (x,y).
     returns an overlay of a distribution over positions on the pacman
     board representing an agent's beliefs about the positions of each
     agent.
  
  NOTE: Since the opposing agents' positions are not given (i.e. not
  directly observable), a joint particle abstraction should be used.
  *********************** Joint Particle Junk ***********************************************************
  def initialize(self, gameState, legalPositions, numParticles = 600):
    "Stores information about the game, then initializes particles."
    self.numGhosts = gameState.getNumAgents() - 1
    self.numParticles = numParticles
    self.ghostAgents = []
    self.legalPositions = legalPositions
    self.initializeParticles()
    
  def initializeParticles(self):
    "Initializes particles randomly.  Each particle is a tuple of ghost positions."
    self.particles = []
    for i in range(self.numParticles):
      self.particles.append(tuple([random.choice(self.legalPositions) for j in range(self.numGhosts)]))

  def addGhostAgent(self, agent):
    "Each ghost agent is registered separately and stored (in case they are different)."
    self.ghostAgents.append(agent)
    
  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the gameState.

    To loop over the ghosts, use:

      for i in range(self.numGhosts):
        ...

    Then, assuming that "i" refers to the (0-based) index of the
    ghost, to obtain the distributions over new positions for that
    single ghost, given the list (prevGhostPositions) of previous
    positions of ALL of the ghosts, use this line of code:

      newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, prevGhostPositions),
                                                  i + 1, self.ghostAgents[i])

    Note that you may need to replace "prevGhostPositions" with the
    correct name of the variable that you have used to refer to the
    list of the previous positions of all of the ghosts, and you may
    need to replace "i" with the variable you have used to refer to
    the index of the ghost for which you are computing the new
    position distribution.

    As an implementation detail (with which you need not concern
    yourself), the line of code above for obtaining newPosDist makes
    use of two helper functions defined below in this file:

      1) setGhostPositions(gameState, ghostPositions)
          This method alters the gameState by placing the ghosts in the supplied positions.
      
      2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
          This method uses the supplied ghost agent to determine what positions 
          a ghost (ghostIndex) controlled by a particular agent (ghostAgent) 
          will move to in the supplied gameState.  All ghosts
          must first be placed in the gameState using setGhostPositions above.
          Remember: ghosts start at index 1 (Pacman is agent 0).  
          
          The ghost agent you are meant to supply is self.ghostAgents[ghostIndex-1],
          but in this project all ghost agents are always the same.
    """
    newParticles = []
    for oldParticle in self.particles:
      newParticle = list(oldParticle) # A list of ghost positions
      "*** YOUR CODE HERE ***"
      # note that the length of list newParticle is
      # equal to the number of ghosts agents
      for i, pos in enumerate(newParticle):
        updatedState = setGhostPositions(gameState, newParticle)
        newPosDist = getPositionDistributionForGhost(gameState, i+1, self.ghostAgents[i])
        newParticle[i] = util.sample(newPosDist)
      newParticles.append(tuple(newParticle))
    self.particles = newParticles
  
  def observeState(self, gameState):
    """
    Resamples the set of particles using the likelihood of the noisy observations.

    As in elapseTime, to loop over the ghosts, use:

      for i in range(self.numGhosts):
        ...

    A correct implementation will handle two special cases:
      1) When a ghost is captured by Pacman, all particles should be updated so
        that the ghost appears in its prison cell, position (2 * i + 1, 1),
        where "i" is the 0-based index of the ghost.

        You can check if a ghost has been captured by Pacman by
        checking if it has a noisyDistance of 999 (a noisy distance
        of 999 will be returned if, and only if, the ghost is
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
          # logging.warning('self.particles does not reflect captured ghosts')
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
      
  def getPrisonParticles(self, particles, ghostIndex):
    """
    Updates all associated particles with the ith ghost to
    its respective prison cell given position (2 * i + 1, 1)
    """
    p = list(particles)
    p[ghostIndex] = ( 2 * ghostIndex + 1, 1)
    return tuple(p)
    
  def getBeliefDistribution(self):
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    return dist

  # One JointInference module is shared globally across instances of MarginalInference 
  jointInference = JointParticleFilter()

  def getPositionDistributionForGhost(gameState, ghostIndex, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied gameState.
    """
    ghostPosition = gameState.getGhostPosition(ghostIndex) 
    actionDist = agent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist
    
  def setGhostPositions(gameState, ghostPositions):
    "Sets the position of all ghosts to the values in ghostPositionTuple."
    for index, pos in enumerate(ghostPositions):
      conf = game.Configuration(pos, game.Directions.STOP)
      gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
    return gameState  

  ****************************** END OF JOINT PARTICLE JUNK *********************************************
  '''
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}
  
class PrimaryAgent(BaseAgent):
  """
  A qlearning agent that seeks food.
  
  *** IDEA: use combination of P3 - Question 9 and imported/exported JSON data
  ***       - for training phase, utilize greedy agent from P4
  ***       - readjust weights  for q learning in execution phase
  ***         s.t. learning rate starts off at its largest in initial
  ***         game and descalates rapidly
  ***         learning rate = 1/(i**2), where i is index of games n, 1 <= i <= n
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class SecondaryAgent(BaseAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  
  ''' Assuming that a joint particle approach is the best for a defensive agent,
      the following is the basis for that (taken directly from p4):
      


  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

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
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

