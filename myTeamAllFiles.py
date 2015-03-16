# Gerald Brown (gemabrow@ucsc.edu) & Alfred Young (ayoung4@ucsc.edu) // CMPS 140 -- Winter 2015
# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
from captureAgents import CaptureAgent
from game import Directions, Actions, Agent
from util import nearestPoint
import game
import layout
#import inference
#import featureExtractor
import cPickle as pickle
import random, time, util, pprint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'BaseAgent', second = 'EphemeralAgent', **args):
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
  #random.seed('R1ckR011d')
  #agents = [first, second]
  #first = random.choice(agents)
  #agents.remove(first)
  #second = agents.pop()
  
  # The following line is an example only; feel free to change it.
  return [BaseAgent(firstIndex), EphemeralAgent(secondIndex)]

##########
# Agents #
##########
class BaseAgent(CaptureAgent):

  def __init__( self, index, timeForComputing = .1, extractor = "MasterExtractor", **args):
    CaptureAgent.__init__(self, index, timeForComputing)
    self.enemyBeliefs = util.Counter()
    # try reinitializing weights
    # to values from prior bouts
    #print type(self.weights)
    self.weights = util.Counter()
    self.QValues = util.Counter()
    # setting indices for team and opponents
    self.friendIndex = self.index + 2
    if self.friendIndex > 3:
      self.friendIndex = self.friendIndex % 2
    self.registerTeam([self.index, self.friendIndex])
    self.enemyIndices = [ number for number in range(0, 4) if number not in self.agentsOnTeam ]
    self.inferenceType = ExactInference
    self.inferenceModules = [ self.inferenceType( index, self ) 
                             for index in self.enemyIndices ]
    self.weightsFilename = str(extractor)+str(self.index)+'.weights'
    extractorType = MasterExtractor#util.lookup(extractor, globals())
    self.featExtractor = MasterExtractor(self)
    try:
      with open(self.weightsFilename, 'rb') as infile:
        self.weights = pickle.load(infile)
    except (IOError, EOFError):
      print "No file '"+self.weightsFilename+" exists."
    
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
    ########################### inference initialization #############################################
    for inference in self.inferenceModules: inference.initialize(gameState)
    self.enemyBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
    self.firstMove = True
    ##################################################################################################
    self.walls = gameState.getWalls()
    
  def observationFunction(self, gameState):
    return gameState.makeObservation(self.index)
  
  def getAction(self, gameState):
    """
    Calls to update beliefs, then chooses an action
    """
    # Append current gameState to observation history
    self.observationHistory.append(gameState)
    # Updates beliefs
    self.updateBeliefs(gameState)
    return self.chooseAction(gameState)
  
  def updateBeliefs(self, gameState):
    """
    Updates self.enemyBeliefs
    """
    for index, inf in enumerate(self.inferenceModules):
      if not self.firstMove: inf.elapseTime(gameState)
      self.firstMove = False
      observation = gameState.getAgentDistances()
      inf.observe(observation, gameState)
      self.enemyBeliefs[index] = inf.getBeliefDistribution()
      
  # NOTE: MOST IMPORTANT function to override
  def chooseAction(self, gameState):
    """
    Gets the most likely positions of each enemy and returns an
    action to get closer to the closest enemy
    chooseAction, depending on the agent's state will take the
    action or discard it
    """
    legal = gameState.getLegalActions(self.index)
    pos = gameState.getAgentPosition(self.index)
    successorPosition = [ ( Actions.getSuccessor(pos, a), a ) for a in legal ]
    bestActions = util.PriorityQueue()
    scared = gameState.data.agentStates[self.index].scaredTimer > 0
    for pos, a in successorPosition:
      x, y = pos
      x = int(x)
      y = int(y)
      pos = x, y
      bestActions.push( (pos, a), (-1 * self.evaluate(gameState, a)) )
      
    return bestActions.pop()[1]
  
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    QValue = 0.0
    # extract feature vectors
    featureVectors = self.featExtractor.getFeatures(gameState, action)
    # perform dotProduct multiplication
    for fV in featureVectors:
      QValue += featureVectors[fV]
    return QValue
  
  def getDistribution(self):
    """
    Returns the distribution from beliefs for each enemy in list form
    """
    enemyDistribution = [ enemyBelief for enemyBelief 
                         in self.enemyBeliefs ]
    self.displayDistributionsOverPositions(enemyDistribution)
    return enemyDistribution
  
  def getEnemyPositions(self, gameState):
    """
    From a distribution and what is observable in the current gamestate, 
    returns a dict of the most probable positions of each enemy agent
    """
    enemyPositions = {}
    for i, inf in enumerate(self.inferenceModules):
      enemyPositions[self.enemyIndices[i]] = inf.mostProbablePosition()
    return enemyPositions
  
  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = self.featExtractor.getFeatures(gameState, action)
    return features
  
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
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
  *** TODO: Double check efficacy of passing in self to featExtractor
  ***       i.e. featExtractor.getFeatures(state, self, action)
  """
  from capture import CaptureRules
  
  def __init__( self, index, timeForComputing = .1, 
               alpha = 0.2, epsilon = 0.05, gamma = 0.8, numTraining = 100, **args):
    self.episodesSoFar = 0
    self.accumTrainRewards = 0.0
    self.accumTestRewards = 0.0
    self.alpha = float(alpha)
    self.epsilon = float(epsilon)
    self.discount = float(gamma)
    self.numTraining = int(numTraining)
    BaseAgent.__init__( self, index, timeForComputing, **args )
    
  def registerInitialState(self, gameState):
    """
    Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    BaseAgent.registerInitialState(self, gameState)
    self.startEpisode()
    if self.episodesSoFar == 0:
        print 'Beginning %d episodes of Training' % (self.numTraining)

  def getValue(self, gameState):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return 0.0

    # creates a list of all Q Values for legal actions from current state
    qValues = [self.getQValue(gameState, action) for action in gameState.getLegalActions(self.index)]
    # returns the max from aforementioned list
    #print max(qValues)
    return max(qValues)
      
  def getQValue(self, gameState, action):
    """
      Should return Q(gameState,action) = w * featureVector
      where * is the dotProduct operator
    """
    QValue = 0.0
    # extract feature vectors
    featureVectors = self.featExtractor.getFeatures(gameState, action)
    # perform dotProduct multiplication
    for fV in featureVectors:
      QValue += featureVectors[fV] * self.weights[fV]
      #print self.weights[fV]
    return QValue

  def getPolicy(self, gameState):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    # intialize policy to None
    policy = None
    legalActions = gameState.getLegalActions(self.index)
    # if there are no legal actions, return policy (s.t. policy = None)
    if len(legalActions) == 0:
      return policy
    
    basePolicy = BaseAgent.chooseAction(self, gameState)
    bestValue = self.getValue(gameState)
    #print "bestValue: {}".format(bestValue)
    bestActions = []
    bestActions.append(basePolicy)
    for action in legalActions:
      # access QValue in this way due to "Important" note in getQValue
      thisValue = self.getQValue(gameState, action)
      # if the value matches that of the best action, append
      # NOTE: since there may be multiple actions that have
      # the "best value" to them, we append all actions
      # that share this attribute
      if thisValue == bestValue:
        #print "appended action"
        bestActions.append(action)
    
    # choose a random action from the list of actions
    # associated with the best value
    if bestActions:
      policy = random.choice(bestActions)

    return policy

  def chooseAction(self, gameState):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    
    return action
    """
    # Append current gameState to observation history
    self.observationHistory.append(gameState)
    # get legal actions, initialize returned action to None
    legalActions = gameState.getLegalActions(self.index)
    action = None
    
    # if there are no legal actions, return action (s.t. action = None)
    if len(legalActions) == 0:
      #print "no legal actions!"
      return action
    
    # Updates beliefs
    self.updateBeliefs(gameState)
    # "With probability self.epsilon, we should take a random action..."
    if util.flipCoin(self.epsilon):
      action = random.choice(legalActions)
    # "...and take the best policy action otherwise."
    else:
      action = self.getPolicy(gameState)
      
    self.doAction(gameState,action)
    return action
  
  def doAction(self, gameState, action):
    self.lastState = gameState
    self.lastAction = action
    
  def update(self, gameState, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    # correction = ( R(s,a) + gamma * V(s') ) - Q(s,a)
    # changes the learning factor such that it is more extreme
    # towards the beginning of a round and levels out over time
    # reinitializes denominator of alpha if time args passed in
    #self.alphaNum -= 1
    #self.alpha = float(self.alphaNum/self.alphaDen)
    correction = reward + self.discount * self.getValue(nextState) - self.getQValue(gameState, action)
    
    featureVectors = self.featExtractor.getFeatures(gameState, action)
    for fV in featureVectors:
      # w_i <- w_i + alpha * [correction] * f_i(s,a)
      self.weights[fV] += self.alpha * correction * featureVectors[fV]
      
  def observeTransition(self, gameState,action,nextState,deltaReward):
    """
        Called by environment to inform agent that a transition has
        been observed. This will result in a call to self.update
        on the same arguments
    """
    self.episodeRewards += deltaReward
    self.update(gameState,action,nextState,deltaReward)

  def startEpisode(self):
    """
    Called by environment when new episode is starting
    """
    self.lastState = None
    self.lastAction = None
    self.episodeRewards = 0.0

  def stopEpisode(self):
    """
      Called by environment when episode is done
    """
    if self.episodesSoFar < self.numTraining:
      self.accumTrainRewards += self.episodeRewards
    else:
      self.accumTestRewards += self.episodeRewards
    self.episodesSoFar += 1
    if self.episodesSoFar >= self.numTraining:
      #print "no training"
      # Take off the training wheels
      self.epsilon = 0.0    # no exploration
      self.alpha = 0.0      # no learning

  def isInTraining(self):
    # print "*********************************TRAINING ****************************************************"
    return self.episodesSoFar < self.numTraining

  def isInTesting(self):
    return not self.isInTraining()
    
  def observationFunction(self, gameState):
    """
        This is where we ended up after our last action.
        The simulation should somehow ensure this is called
    """
    if not self.lastState is None:
        reward = self.getScore(gameState) - self.getScore(self.lastState)
        self.observeTransition(self.lastState, self.lastAction, gameState, reward)
    observedState = gameState.makeObservation(self.index)
    observedState.data.timeleft = gameState.data.timeleft - 1
    return observedState

  def final(self, gameState):
    """
      Called by Pacman game at the terminal gameState
    """
    self.lastState = self.getPreviousObservation()
    self.gameState = self.getCurrentObservation()
    dx, dy = self.gameState.getAgentPosition(self.index)
    x, y = self.lastState.getAgentPosition(self.index)
    vector = (dx-x, dy-y)
    self.lastAction = Actions.vectorToDirection(vector)
    deltaReward = self.getScore(self.gameState) * float( 1200/(gameState.data.timeleft+1) )
    
    self.observeTransition(self.getPreviousObservation(), self.lastAction, 
                           self.getCurrentObservation(), deltaReward)
    self.stopEpisode()

    # Make sure we have this var
    if not 'episodeStartTime' in self.__dict__:
        self.episodeStartTime = time.time()
    if not 'lastWindowAccumRewards' in self.__dict__:
        self.lastWindowAccumRewards = 0.0
    self.lastWindowAccumRewards += self.getScore(gameState)

    NUM_EPS_UPDATE = 100
    if self.episodesSoFar % NUM_EPS_UPDATE == 0:
        print 'Reinforcement Learning Status:'
        windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
        if self.episodesSoFar <= self.numTraining:
            trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
            print '\tCompleted %d out of %d training episodes' % (
                   self.episodesSoFar,self.numTraining)
            print '\tAverage Rewards over all training: %.2f' % (
                    trainAvg)
        else:
            testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
            print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
            print '\tAverage Rewards over testing: %.2f' % testAvg
        print '\tAverage Rewards for last %d episodes: %.2f'  % (
                NUM_EPS_UPDATE,windowAvg)
        print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
        self.lastWindowAccumRewards = 0.0
        self.episodeStartTime = time.time()
        
    # Where we save our accumulated weights so far
    if self.episodesSoFar % 5 == 0 or self.episodesSoFar == self.numTraining:
        try:
          with open(self.weightsFilename, 'wb') as outfile:
            pickle.dump(self.weights, outfile, protocol=pickle.HIGHEST_PROTOCOL)
          print "updated weights"
        except IOError:
          print "Unable to write file "+self.weightsFilename

class inference:
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

class ExactInference(inference):
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


NO_DBZ = 1 # No division by zero -- not to confused with those against the DragonBall Z series

class MasterExtractor:
  # NOTE: gameState.getAgentState(index) = gameState.data.agentStates[index] 
  #       could hold exact position of opposing team
  # OR use starting position plus AgentState.getDirection & vector 
  #    difference to figure new position
  """
  Returns features for a qLearning agent in Capture-the-Flag:
  - what team the agent is on
  - initial position of agent
  - initial position of enemy agents
  - position of agent
  - agent's state
  - agent's scared timer
  - position of teammate
  - teammate's state
  - whether food will be eaten
  - how far away the next food is
  - how far away the nearest enemy is (actual distance, if available, or noisy distance)
  - the state of each enemy
  - whether an enemy collision is imminent
  - whether an enemy is one step away
  """
  def __init__(self, myAgent):
    # passing in agent should give us access to beliefs
    self.agent = myAgent
  
  def comparePrevState(self, gameState, fn):
    # NOTE: assumes comparison will be of matrices
    prevState = self.agent.getPreviousObservation()
    currStatus = fn(gameState)
    priorStatus = currStatus
    if prevState is not None:
      priorStatus = fn(prevState)
    return matrixDiff(priorStatus, currStatus)
  
  def getFeatures(self, gameState, action):
    successor = gameState.generateSuccessor(self.agent.index, action)
    walls = gameState.getWalls()
    denom = (walls.width * walls.height) 
    features = util.Counter()
    # ***************** features of agents ***********************
    initialPos = gameState.getInitialAgentPosition(self.agent.index)
    prevAgentState = gameState.getAgentState(self.agent.index)
    prevPos = prevAgentState.getPosition()
    agentState = successor.getAgentState(self.agent.index)
    nextX, nextY = agentState.getPosition()
    myPos = int(nextX), int(nextY)
    scaredTime = agentState.scaredTimer
    
    friendPos = gameState.getAgentPosition(self.agent.friendIndex)
    halfWidth = walls.width / 2
    halfHeight = walls.height / 2

    atHome = lambda xPos, isRed: True if (isRed and xPos < halfWidth) or (not isRed and xPos > halfWidth) else False
    topHalf = lambda yPos: True if yPos > halfHeight else False
    self.enemyPositions = self.agent.getEnemyPositions(gameState)
    enemies = []
    for enemyIndex in self.agent.enemyIndices:
      enemy = {}
      enemy['index'] = enemyIndex
      enemy['pos'] = self.enemyPositions[enemyIndex]
      enemy['isPacman'] = gameState.getAgentState(enemyIndex).isPacman
      enemy['scaredTimer'] = gameState.getAgentState(enemyIndex).scaredTimer
      enemies.append(enemy)
    
    # ************************* game features **************************************
    
    " LIKE A BAT OUTTA HELL "
    if myPos == prevPos and myPos in Actions.getLegalNeighbors(prevPos, walls):
      features['camping-penalty'] -= 1
      #print features['camping-penalty']


    " WHAT'RE MY OPTIONS, HMMMM??? "
    features['available-moves-from-successor'] = len(Actions.getLegalNeighbors(myPos, walls))
    #print "successor moves ", features['available-moves-from-successor']
    # trend towards middle
    # and away from the initial spawning point
    
    food = self.agent.getFood(gameState)
    eatFood = food.asList()
    if self.agent.index > self.agent.friendIndex:
      closestFood = min(eatFood, key = lambda food: self.agent.distancer.getDistance(food, prevPos) if topHalf(food[1]) else None)
    else:
      features['engage-enemy-factor'] = 1.25 / (NO_DBZ + len(self.agent.getFoodYouAreDefending(gameState).asList()) )
      closestFood = min(eatFood, key = lambda food: self.agent.distancer.getDistance(food, prevPos) if not topHalf(food[1]) else None)
    if closestFood is None:
      closestFood = min(eatFood, key = lambda food: self.agent.distancer.getDistance(food, prevPos) )
    
    meToFood = self.agent.distancer.getDistance(closestFood, myPos)
    meToFoodPrev = self.agent.distancer.getDistance(closestFood, prevPos)
    
    if meToFood < meToFoodPrev:
      features['food-factor'] = float( 100 / (NO_DBZ + len(eatFood)) )
    
    closestEnemy = min(enemies, key = lambda enemy: self.agent.distancer.getDistance(myPos, enemy['pos']))
    enemyDistance = self.agent.distancer.getDistance(myPos, closestEnemy['pos'])
    enemyToFood = self.agent.distancer.getDistance(closestFood, closestEnemy['pos'])
    features['engage-enemy-factor'] += 1 / (NO_DBZ +  2.5 * abs(myPos[0] - initialPos[0]) + enemyDistance )
    
    "TIME TO PLAY SOME D"
    if atHome(nextX, self.agent.red):
      invaders = [enemy for enemy in enemies if enemy['isPacman']]
      closeInvaders = sum(myPos in Actions.getLegalNeighbors(i['pos'], walls) for i in invaders)
      if closeInvaders > 0 and scaredTime == 0:
        features["#-of-invaders-1-step-away"] = closeInvaders
        closestInvader = min(invaders, key = lambda enemy: self.agent.distancer.getDistance(myPos, enemy['pos']))
        if myPos == closestInvader['pos']:
          features['ate-invader'] = 1000.0 
        elif myPos in Actions.getLegalNeighbors(closestInvader['pos'], walls):
          features['pursue-invader'] = 100.0
      elif scaredTime >= 1:
        distanceToCenter = min(abs(legalPos[0] - halfWidth) for legalPos in Actions.getLegalNeighbors(myPos, walls))
        features['FLEE'] = (10.0 / (NO_DBZ + distanceToCenter ))
    
    "EAT EM UP"
    while not atHome(nextX, self.agent.red):
      if closestFood in Actions.getLegalNeighbors(myPos, walls):
        features['food-factor'] +=  10 if closestFood == myPos else features['food-factor'] + 5
      if meToFood < enemyToFood or enemyDistance < meToFood:
        features['food-factor'] += 2
      ghosts = [enemy for enemy in enemies if enemy['isPacman']]
      if ghosts:
        closestGhost = min(ghosts, key = lambda enemy: self.agent.distancer.getDistance(enemy['pos'], myPos))
        if closestGhost['scaredTimer'] >= 1:
          # TIME TO RAGE
          # #print "STRAIGHT RAGIN'", myPos
          features['RAGE-RAGE-RAGE'] = 666.0/( NO_DBZ + (min(enemyDistance, meToFood)) )
          features['RAGE-CHOMP'] = 666.0 if closestGhost['pos'] == myPos else 1.0
          features['RAGE-CHOMP'] += 666.1 if closestFood == myPos else 1.5
          break
        
        closeGhosts = sum(myPos in Actions.getLegalNeighbors(g['pos'], walls) for g in ghosts)
        try:
          eatCapsules = self.agent.getCapsules(gameState)
          if len(eatCapsules) > 0:
            closestCapsule = min(eatCapsules, key = lambda capsule: self.agent.distancer.getDistance(myPos, capsule))
            enemyToCapsule = self.agent.distancer.getDistance(closestCapsule, closestGhost['pos'])
            meToCapsule = self.agent.distancer.getDistance(myPos, closestCapsule)
            if meToCapsule <= enemyToCapsule:
              features['capsule-craving'] = float( 1/ (NO_DBZ + (enemyDistance*meToCapsule) ) )
              features['eat-capsule'] = 1.0 if closestCapsule == myPos else 0.0
        except (IndexError, ValueError):
          pass
        if closeGhosts > 0 and len(Actions.getLegalNeighbors(prevPos, walls)) == 2:
          # means our only options are stop or eat it (by it, we mean the ghost)
          features["suicide-pill"] = 100
          break
        
      else:
        features['food-factor'] += 5
      break
    return features
  

