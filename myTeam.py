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
import inference
import featureExtractor
import cPickle as pickle
import random, time, util, pprint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'BaseAgent', second = 'BaseAgent', **args):
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
        directly observable)
  '''
  def __init__( self, index, timeForComputing = .1, extractor = "MasterExtractor", **args):
    CaptureAgent.__init__(self, index, timeForComputing)
    print self.index
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
    self.inferenceType = inference.ExactInference
    self.inferenceModules = [ self.inferenceType( index, self ) 
                             for index in self.enemyIndices ]
    self.weightsFilename = str(extractor)+str(self.index)+'.weights'
    extractorType = util.lookup(extractor, globals())
    self.featExtractor = extractorType(self)
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
    for pos, a in successorPosition:
      x, y = pos
      x = int(x)
      y = int(y)
      pos = x, y
      print self.evaluate(gameState, a)
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
               alpha = 0.2, epsilon = 0.02, gamma = 0.8, numTraining = 50, **args):
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
    #if self.episodesSoFar == 0:
        #print 'Beginning %d episodes of Training' % (self.numTraining)

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

    bestValue = self.getValue(gameState)
    #print "bestValue: {}".format(bestValue)
    bestActions = []
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
    else:
      print "bad bad bad"
      policy = random.choice(legalActions)

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
      print "no legal actions!"
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
      print "no training"
      # Take off the training wheels
      #self.epsilon = 0.0    # no exploration
      #self.alpha = 0.0      # no learning

  def isInTraining(self):
    print "********************************* IN TRAINING ********************************"
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
    deltaReward = self.getScore(self.gameState) #AGGRESSIVE DELTA * float( 1200/(gameState.data.timeleft+1) )
    
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
    if self.episodesSoFar <= self.numTraining:
        try:
          with open(self.weightsFilename, 'wb') as outfile:
            pickle.dump(self.weights, outfile, protocol=pickle.HIGHEST_PROTOCOL)
          print "updated weights"
        except IOError:
          print "Unable to write file "+self.weightsFilename