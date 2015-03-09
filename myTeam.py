# Gerald Brown (gemabrow@ucsc.edu) & Alfred Young (ayoung4@ucsc.edu) // CMPS 140 -- Winter 2015
# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from game import Directions, Actions
from util import nearestPoint
import game
import inference
import qLearningAgent # would like to instantiate directly
import featureExtractor
import distanceCalculator
import random, time, util, json

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'EphemeralAgent', second = 'AnotherAgent'):
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
  
  # randomize which agent is first or second,
  # just to mix things up
  random.seed(R1ckR011d)
  agents = [first, second]
  first = random.choice(agents)
  second = agent in agents if not first
  
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class BaseAgent(CaptureAgent):
  """
  A base agent to serve as a foundation for the varying agent structures.
  Tracks beliefs about agents' positions.
  """
  ''' 
  INHERITED VARIABLES (from CaptureAgent):
  # index for this agent
  self.index
  
  # true if agent is on the red team, false otherwise
  self.red
  
  # Agent objects controlling this agent's team (including the agent)
  self.agentsOnTeam
  
  # Maze distance calculator
  self.distancer
  
  # history of observations -- a sequential list of gamestates that have occurred in this game
  self.observationHistory
  
  # an amount of time to give each turn for computing maze distances
  self.timeForComputing

  INHERITED FUNCTIONS (from CaptureAgent):
  self.final(gameState):                 resets observationHistory
  
  self.registerTeam(agentsOnTeam):       fills CaptureAgent.agentsOnTeam with indices of agents on team
  
  self.observationFunction(gameState):   return gameState.makeObservation(CaptureAgent.index)
  
  self.getAction(gameState):             appends current gameState on to our observation history
                                         and calls our choose action method
  
  self.getFood(gameState):               returns a matrix with the food we're meant to eat
                                         in the form m[x][y]==true if there is food for us
                                         in that square (based on our team color).
  
  self.getFoodYouAreDefending(gameState):returns the food we should protect in the form
                                         of a matrix m[x][y]==true if there is food our
                                         opponent can eat at those coordinates.
  
  self.getCapsules(gameState):           duh
  
  self.getCapsulesYouAreDefending(gameState): also, duh
  
  self.getOpponents(gameState):          returns agent indices of our opponents in list form.
  
  self.getTeam(gameState):               returns a list of indices of the agents on our team.
  
  self.getScore(gameState):              returns a number that is the difference in teams' scores.
                                         will be negative if we're a bunch of sissy la-la losers.
  
  self.getMazeDistance(pos1, pos2):      returns the maze distance from pos1 to pos2.
  
  self.getPreviousObservation():         returns the last GameState object this agent saw
                                         (may not include the exact locations of our opponent's agents)
  
  self.getCurrentObservation():          like before, but now
  
  self.displayDistributionsOverPositions(distributions):  arg distributions is a tuple or list of util.Counter objects,
                                                          where the i'th Counter has keys that are board positions (x,y)
                                                          and values that encode the probability that agent i is at (x,y).
                                                          returns an overlay of a distribution over positions on the pacman
                                                          board representing an agent's beliefs about the positions of each
                                                          agent.
  
  NOTE: Since the opposing agents' positions are not given (i.e. not
  directly observable), a joint particle abstraction should be used.
  '''
  def __init__( self, inference = "ExactInference"):
    self.inferenceType = util.lookup(inference, globals())
    
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
    self.friendIndex = self.index + 2
    if self.friendIndex > 3:
      self.friendIndex = self.friendIndex % 2
    self.enemyAgents = sorted(self.getOpponents(gameState))
    self.numEnemies = len(self.enemyAgents)
    self.registerTeam([self.index, self.friendIndex])
    # legalPositions may be unnecessary here --> qLearningAgent instead
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]   
    ########################### inference initialization #############################################
    self.inferenceModules = [ self.inferenceType( enemy, self ) 
                             for enemy in self.enemyAgents ]
    for inference in self.inferenceModules: inference.initialize(gameState)
    self.enemyBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
    self.firstMove = True
    ##################################################################################################
    
    # NOTE: Cache of maze distances to every pair of positions
    # can be accessed by an agent with the following:
    # self.distancer.getDistance( pos1, pos2 )
    
  ''' NOTE: Defined in captureAgents, shot not be overridden
  def observationFunction(self, gameState):
    return gameState.makeObservation(self.index)
  '''
  
  def getAction(self, gameState):
    """
    Calls chooseAction on a grid position, but continues on half positions,
    while updating beliefs. Sends resulting update to choose an action based
    on updated beliefs if not in a halfway position.
    """
    self.observationHistory.append(gameState)

    # Updates beliefs
    for index, inf in enumerate(self.inferenceModules):
      if not self.firstMove: inf.elapseTime(gameState)
      self.firstMove = False
      inf.observeState(gameState)
      self.opponentBeliefs[index] = inf.getBeliefDistribution()
    
    # Appends current gameState to observation history
    # and will call chooseAction if in an actual state
    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    if myPos != nearestPoint(myPos):
      # We're halfway from one position to the next
      return gameState.getLegalActions(self.index)[0]
    else:
      return self.chooseAction(gameState)
  
  # NOTE: MOST IMPORTANT function to override
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    legal = [ a for a in gameState.getLegalActions(self.index) ]
    values = [self.evaluate(gameState, a) for a in legal]
    
    #**************************************************************************************
    #********************** may want to change this ***************************************
    #***************** could act as a good defense agent **********************************
    #**************************************************************************************
    # PriorityQueue for maintaining action towards/away closest ghost
    #distanceActionPQ = util.PriorityQueue()
    #
    #if agent is a ghost and not scared
    #if agent is a ghost
    #if agent is pacman
    #|_____\ move all of this to either
    #      / evaluate() or getFeatures()
    #for agentPos, action in successorPosition:
    #  for enemyPos in possibleEnemyPositions:
    #    distanceActionPQ.push( action, self.distancer.getDistance(agentPos, enemyPos) )
    # 
    #return minDistanceActionPQ.pop()
    #**************************************************************************************
    
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    # maxValue = max(values)
    # bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    # return random.choice(bestActions)
    
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
  
  def getDistribution(self, gameState):
    """
    Returns the position distribution for indices
    that are found in the list self.enemyAgents
    """
    enemyPositionDistributions = [beliefs for i,beliefs 
                                  in enumerate(self.enemyBeliefs)
                                  if i in self.enemyAgents]
    
  def getPositions(self, distribution)
    """
    From a distribution, returns a list of the most
    probable positions of each enemy agent
    """
    probableEnemyPositions = [positionDistribution.argMax() 
                             for positionDistribution 
                             in enemyPositionDistributions]
    return probableEnemyPositions
  
  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    self.displayDistributionsOverPositions( self.getDistribution(successor) )
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}
  
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
    
class EphemeralAgent(BaseAgent):
  """
  A qlearning agent.
  
  *** IDEA: use combination of P3 - Question 9 and imported/exported JSON data
  ***       - for training phase, utilize greedy agent from P4
  ***       - readjust weights  for q learning in execution phase
  ***         s.t. learning rate starts off at its largest in initial
  ***         game and descalates rapidly
  ***         learning rate = 1/(i**2), where i is index of games n, 1 <= i <= n
  """
  from capture import CaptureRules
  
  def __init__(self, alpha=1200, epsilon=0.05, gamma=0.8, numTraining = 10):
    """
    Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    self.alphaNum = float(alpha)
    self.alphaDen = float(alpha)
    self.alpha = float(alphaNum/alphaDen) # alpha denominator is the length of the game
    self.epsilon = float(epsilon)
    self.discount = float(gamma)
    self.numTraining = int(numTraining)
    self.featExtractor = MasterExtractor()
    # initialize the weights to be assigned to varying features
    self.weights = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    QValue = 0.0
    # extract feature vectors
    featureVectors = self.featExtractor.getFeatures(state, self, action)
    # perform dotProduct multiplication
    for fV in featureVectors:
      QValue += featureVectors[fV] * self.weights[fV]
    return QValue

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    # correction = ( R(s,a) + gamma * V(s') ) - Q(s,a)
    # changes the learning factor such that it is more extreme
    # towards the beginning of a round and levels out over time
    # reinitializes denominator of alpha if time args passed in
    if state.data.timeleft > self.alphaDen:
      self.alphaDen = state.data.timeleft
    self.alphaNum = state.data.timeleft
    self.alpha = float(alphaNum/alphaDen)
    correction = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
    
    featureVectors = self.featExtractor.getFeatures(state, self, action)
    for fV in featureVectors:
      # w_i <- w_i + alpha * [correction] * f_i(s,a)
      self.weights[fV] += self.alpha * correction * featureVectors[fV]
      
  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    legalActions = self.getLegalActions(state)
    if len(legalActions) == 0:
      return 0.0

    # creates a list of all Q Values for legal actions from current state
    qValues = [self.getQValue(state, action) for action in state.getLegalActions(self.index)]
    # returns the max from aforementioned list
    return max(qValues)

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    # intialize policy to None
    policy = None
    legalActions = state.getLegalActions(self.index)
    # if there are no legal actions, return policy (s.t. policy = None)
    if len(legalActions) == 0:
      return policy
    
    # find the value of the best action
    bestValue = self.getValue(state)
    bestActions = []
    for action in legalActions:
      # access QValue in this way due to "Important" note in getQValue
      thisValue = self.getQValue(state, action)
      # if the value matches that of the best action, append
      # NOTE: since there may be multiple actions that have
      # the "best value" to them, we append all actions
      # that share this attribute
      if thisValue == bestValue:
        bestActions.append(action)
    
    # choose a random action from the list of actions
    # associated with the best value
    policy = random.choice(bestActions)
    return policy

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    # get legal actions, initialize returned action to None
    legalActions = state.getLegalActions(self.index)
    action = None
    
    # if there are no legal actions, return action (s.t. action = None)
    if len(legalActions) == 0:
      return action
    
    # "With probability self.epsilon, we should take a random action..."
    if util.flipCoin(self.epsilon):
      action = random.choice(legalActions)
    # "...and take the best policy action otherwise."
    else:
      action = self.getPolicy(state)
      
    self.doAction(state,action)
    return action

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    #**********add super-class final method************************************
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      print json.dump(self.weights)
      pass

#******************************************************************************************************************************************************************
#*************************************************SECONDARY AGENT**************************************************************************************************
#******************************************************************************************************************************************************************

class AnotherAgent(BaseAgent):
  """
  A secondary agent, playing and reacting to PrimaryAgent's
  most recent action taken to achieve the best cooperative outcome
  """

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

#*****************************************************END OF SECONDARY AGENT***************************************************************************************
#*****************************************************Joint Particle Functions*************************************************************************************
#******************************************************************************************************************************************************************



#******************************************************************************************************************************************************************
#******************************************************FEATURE EXTRACTOR BUSINESS**********************************************************************************
#******************************************************************************************************************************************************************

#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************
#********************************************* POTENTIAL REPRESENTATIONS OF ENEMY AGENTS **************************************************************************
class EnemyAgent( Agent ):
  def __init__( self, index ):
    self.index = index

  def getAction( self, state ):
    dist = self.getDistribution(state)
    if len(dist) == 0: 
      return Directions.STOP
    else:
      return util.chooseFromDistribution( dist )
    
  def getDistribution(self, state):
    "Returns a Counter encoding a distribution over actions from the provided state."
    util.raiseNotDefined()

class DirectionalGhost( GhostAgent ):
  "A ghost that prefers to rush Pacman, or flee when scared."
  def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
    self.index = index
    self.prob_attack = prob_attack
    self.prob_scaredFlee = prob_scaredFlee
  
  def getAction(self, state):
    dist = self.getDistribution(state)
  def getDistribution( self, state ):
    # Read variables from state
    ghostState = state.getGhostState( self.index )
    legalActions = state.getLegalActions( self.index )
    pos = state.getGhostPosition( self.index )
    isScared = ghostState.scaredTimer > 0
    
    speed = 1
    if isScared: speed = 0.5
    
    actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
    newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
    pacmanPosition = state.getPacmanPosition()

    # Select best actions given the state
    distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
    if isScared:
      bestScore = max( distancesToPacman )
      bestProb = self.prob_scaredFlee
    else:
      bestScore = min( distancesToPacman )
      bestProb = self.prob_attack
    bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]
    
    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
    dist.normalize()
    return dist
#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************
#******************************************************************************************************************************************************************
#******************************************************************INFERENCE MODULE BUSINESS***********************************************************************

