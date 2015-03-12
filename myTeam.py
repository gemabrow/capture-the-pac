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
import game
import layout
import inference
import featureExtractor
import random, time, util, json

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'BaseAgent', second = 'BaseAgent'):
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
  random.seed('R1ckR011d')
  agents = [first, second]
  first = random.choice(agents)
  agents.remove(first)
  second = agents.pop()
  
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
  def __init__( self, index, timeForComputing = .1):
    CaptureAgent.__init__(self, index, timeForComputing)
    self.QValues = util.Counter()
    self.weights = util.Counter()
    self.enemyBeliefs = util.Counter()
    # try reinitializing qvalues and weights
    # to ascertained values from prior bouts
    try:
      with open('qValues') as infile:
        json_qvalues = json.load(infile)
        self.QValues = json_qvalues
    except IOError:
      print "No file 'qValues' exists."
    try:
      with open('weights') as infile:
        json_weights = json.load(infile)
        self.weights = json_weights
    except IOError:
      print "No file 'weights' exists."
    
    # setting indices for team and opponents
    self.friendIndex = self.index + 2
    if self.friendIndex > 3:
      self.friendIndex = self.friendIndex % 2
    self.registerTeam([self.index, self.friendIndex])
    self.enemyIndices = [ number for number in range(0, 4) if number not in self.agentsOnTeam ]
    self.inferenceType = inference.ExactInference
    self.inferenceModules = [ self.inferenceType( index, self ) 
                             for index in self.enemyIndices ]
    self.featExtractor = featureExtractor.MasterExtractor(self)

    # ONLY returns the INITIAL agent position, just as gameState.getInitialAgentPosition(index) does
    # self.enemyPositions = dict(  ( index, defaultMap.AgentPosition[index][1] ) for index in self.enemyIndices  )
    
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
    print "Red Team: ", gameState.getRedTeamIndices()
    ########################### inference initialization #############################################
    for inference in self.inferenceModules: inference.initialize(gameState)
    for i, inf in enumerate(self.inferenceModules):
      self.enemyBeliefs[self.enemyIndices[i]] = inf.getBeliefDistribution()
    self.firstMove = True
    ##################################################################################################
    
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
      inf.observeState(gameState)
      self.enemyBeliefs[self.enemyIndices[index]] = inf.getBeliefDistribution()
      
  # NOTE: MOST IMPORTANT function to override
  def chooseAction(self, gameState):
    """
    Gets the most likely positions of each enemy and returns an
    action to get closer to the closest enemy
    chooseAction, depending on the agent's state will take the
    action or discard it
    """
    legal = [ a for a in gameState.getLegalActions(self.index) ]
    values = [(self.evaluate(gameState, a), a) for a in legal]

    enemyPositions = self.getEnemyPositions(gameState)
    # debugging inference---------------------------------------------------------<<<<<<<<<<<<
    self.displayDistributionsOverPositions( self.getDistribution(gameState) )
    valueActions = [(self.evaluate(gameState, a), a) for a in legalActions]
    maxValue = max(valueActions)[0]
    bestActions = [a for v, a in valueActions if v == maxValue]
    return random.choice(bestActions)
    
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
  
  def getDistribution(self):
    """
    Returns the distribution from beliefs for each enemy in list form
    """
    enemyDistribution = [ enemyBelief for enemyBelief in self.enemyBeliefs ]
    return enemyDistribution
  
  def getEnemyPositions(self, gameState):
    """
    From a distribution and what is observable in the current gamestate, 
    returns a dict of the most probable positions of each enemy agent
    """
    enemyPositions = {}
    myPosition  = gameState.getAgentPosition(self.index)
    legalActions = [a for a in gameState.getLegalActions(self.index)]
    # potential successor positions for our agent
    successorPos = [ ( Actions.getSuccessor(myPosition, a), a) for a in legalActions ]
    distribution = self.getDistribution()
    for i, enemyIndex in enumerate(self.enemyIndices):
      enemyPos = gameState.getAgentPosition(enemyIndex)
      enemyPositions[enemyIndex] = enemyPos if enemyPos != None else distribution.argMax()[i]
    return enemyPositions
  
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
  
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    return successor
    
#class EphemeralAgent(BaseAgent):
  #"""
  #A qlearning agent.
  
  #*** IDEA: use combination of P3 - Question 9 and imported/exported JSON data
  #***       - for training phase, utilize greedy agent from P4
  #***       - readjust weights  for q learning in execution phase
  #***         s.t. learning rate starts off at its largest in initial
  #***         game and descalates rapidly
  #***         learning rate = 1/(i**2), where i is index of games n, 1 <= i <= n
  #*** TODO: Double check efficacy of passing in self to featExtractor
  #***       i.e. featExtractor.getFeatures(state, self, action)
  #"""
  #from capture import CaptureRules
  
  #def __init__( self, index, timeForComputing = .1, 
               #alpha = 1200, epsilon = 0.05, gamma = 0.8, numTraining = 10):
    #BaseAgent.__init__( self, index, timeForComputing )
    #self.alphaNum = float(alpha)
    #self.alphaDen = float(alpha)
    #self.alpha = 1.0
    #self.epsilon = float(epsilon)
    #self.discount = float(gamma)
    #self.numTraining = int(numTraining)
    #self.episodesSoFar = 0
    #self.accumTrainRewards = 0.0
    #self.accumTestRewards = 0.0
    
  #def registerInitialState(self, gameState):
    #"""
    #Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
    #alpha    - learning rate
    #epsilon  - exploration rate
    #gamma    - discount factor
    #numTraining - number of training episodes, i.e. no learning after these many episodes
    #"""
    #BaseAgent.registerInitialState(self, gameState)
    #self.startEpisode()
    #if self.episodesSoFar == 0:
        #print 'Beginning %d episodes of Training' % (self.numTraining)

  #def getValue(self, state):
    #"""
      #Returns max_action Q(state,action)
      #where the max is over legal actions.  Note that if
      #there are no legal actions, which is the case at the
      #terminal state, you should return a value of 0.0.
    #"""
    #legalActions = state.getLegalActions(self.index)
    #if len(legalActions) == 0:
      #return 0.0

    ## creates a list of all Q Values for legal actions from current state
    #qValues = [self.getQValue(state, action) for action in state.getLegalActions(self.index)]
    ## returns the max from aforementioned list
    #return max(qValues)
      
  #def getQValue(self, state, action):
    #"""
      #Should return Q(state,action) = w * featureVector
      #where * is the dotProduct operator
    #"""
    #QValue = 0.0
    ## extract feature vectors
    #featureVectors = self.featExtractor.getFeatures(state, action)
    ## perform dotProduct multiplication
    #for fV in featureVectors:
      #QValue += featureVectors[fV] * self.weights[fV]
    #return QValue

  #def getPolicy(self, state):
    #"""
      #Compute the best action to take in a state.  Note that if there
      #are no legal actions, which is the case at the terminal state,
      #you should return None.
    #"""
    ## intialize policy to None
    #policy = None
    #legalActions = state.getLegalActions(self.index)
    ## if there are no legal actions, return policy (s.t. policy = None)
    #if len(legalActions) == 0:
      #return policy
    
    ## find the value of the best action
    #bestValue = self.getValue(state)
    #bestActions = []
    #for action in legalActions:
      ## access QValue in this way due to "Important" note in getQValue
      #thisValue = self.getQValue(state, action)
      ## if the value matches that of the best action, append
      ## NOTE: since there may be multiple actions that have
      ## the "best value" to them, we append all actions
      ## that share this attribute
      #if thisValue == bestValue:
        #bestActions.append(action)
    
    ## choose a random action from the list of actions
    ## associated with the best value
    #policy = random.choice(bestActions)
    #return policy

  #def getAction(self, state):
    #"""
    #Simply calls the getAction method of QLearningAgent and then
    #informs parent of action for Pacman.  Do not change or remove this
    #method.
    
    #action = QlearningAgent.getAction(self,state)
    #self.doAction(state,action)
    #return action
    #"""
    ## get legal actions, initialize returned action to None
    #legalActions = state.getLegalActions(self.index)
    #action = None
    
    ## if there are no legal actions, return action (s.t. action = None)
    #if len(legalActions) == 0:
      #return action
    
    ## "With probability self.epsilon, we should take a random action..."
    #if util.flipCoin(self.epsilon):
      #action = random.choice(legalActions)
    ## "...and take the best policy action otherwise."
    #else:
      #action = self.getPolicy(state)
      
    #return action

  #def update(self, state, action, nextState, reward):
    #"""
       #Should update your weights based on transition
    #"""
    ## correction = ( R(s,a) + gamma * V(s') ) - Q(s,a)
    ## changes the learning factor such that it is more extreme
    ## towards the beginning of a round and levels out over time
    ## reinitializes denominator of alpha if time args passed in
    #if state.data.timeleft > self.alphaDen:
      #self.alphaDen = state.data.timeleft
    #self.alphaNum = state.data.timeleft
    #self.alpha = float(alphaNum/alphaDen)
    #correction = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
    
    #featureVectors = self.featExtractor.getFeatures(state, action)
    #for fV in featureVectors:
      ## w_i <- w_i + alpha * [correction] * f_i(s,a)
      #self.weights[fV] += self.alpha * correction * featureVectors[fV]
      
  #def observeTransition(self, state,action,nextState,deltaReward):
    #"""
        #Called by environment to inform agent that a transition has
        #been observed. This will result in a call to self.update
        #on the same arguments
    #"""
    #self.episodeRewards += deltaReward
    #self.update(state,action,nextState,deltaReward)

  #def startEpisode(self):
    #"""
      #Called by environment when new episode is starting
    #"""
    #self.lastState = None
    #self.lastAction = None
    #self.episodeRewards = 0.0

  #def stopEpisode(self):
    #"""
      #Called by environment when episode is done
    #"""
    #if self.episodesSoFar < self.numTraining:
      #self.accumTrainRewards += self.episodeRewards
    #else:
      #self.accumTestRewards += self.episodeRewards
    #self.episodesSoFar += 1
    #if self.episodesSoFar >= self.numTraining:
      ## Take off the training wheels
      #self.epsilon = 0.0    # no exploration
      #self.alpha = 0.0      # no learning

  #def isInTraining(self):
      #return self.episodesSoFar < self.numTraining

  #def isInTesting(self):
      #return not self.isInTraining()
    
  #def observationFunction(self, state):
    #"""
        #This is where we ended up after our last action.
        #The simulation should somehow ensure this is called
    #"""
    #if not self.lastState is None:
        #reward = state.getScore() - self.lastState.getScore()
        #self.observeTransition(self.lastState, self.lastAction, state, reward)
    #return state

  #def final(self, state):
    #"""
      #Called by Pacman game at the terminal state
    #"""
    #deltaReward = state.getScore() - self.lastState.getScore()
    #self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
    #self.stopEpisode()

    ## Make sure we have this var
    #if not 'episodeStartTime' in self.__dict__:
        #self.episodeStartTime = time.time()
    #if not 'lastWindowAccumRewards' in self.__dict__:
        #self.lastWindowAccumRewards = 0.0
    #self.lastWindowAccumRewards += state.getScore()

    #NUM_EPS_UPDATE = 100
    #if self.episodesSoFar % NUM_EPS_UPDATE == 0:
        #print 'Reinforcement Learning Status:'
        #windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
        #if self.episodesSoFar <= self.numTraining:
            #trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
            #print '\tCompleted %d out of %d training episodes' % (
                   #self.episodesSoFar,self.numTraining)
            #print '\tAverage Rewards over all training: %.2f' % (
                    #trainAvg)
        #else:
            #testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
            #print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
            #print '\tAverage Rewards over testing: %.2f' % testAvg
        #print '\tAverage Rewards for last %d episodes: %.2f'  % (
                #NUM_EPS_UPDATE,windowAvg)
        #print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
        #self.lastWindowAccumRewards = 0.0
        #self.episodeStartTime = time.time()
        
    ## Where we save our accumulated QValues and weights so far
    #if self.episodesSoFar == self.numTraining:
        #with open('weights', 'w') as outfile:
          #json.dump(self.weights, outfile)
        #with open('qValues', 'w') as outfile:
          #json.dump(self.QValues, outfile)

##******************************************************************************************************************************************************************
##*************************************************SECONDARY AGENT**************************************************************************************************
##******************************************************************************************************************************************************************

#class AnotherAgent(BaseAgent):
  #"""
  #A secondary agent, playing and reacting to PrimaryAgent's
  #most recent action taken to achieve the best cooperative outcome
  #"""

  #def getFeatures(self, gameState, action):
    #features = util.Counter()
    #successor = self.getSuccessor(gameState, action)

    #myState = successor.getAgentState(self.index)
    #myPos = myState.getPosition()

    ## Computes whether we're on defense (1) or offense (0)
    #features['onDefense'] = 1
    #if myState.isPacman: features['onDefense'] = 0

    ## Computes distance to invaders we can see
    #enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    #invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    #features['numInvaders'] = len(invaders)
    #if len(invaders) > 0:
      #dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      #features['invaderDistance'] = min(dists)

    #if action == Directions.STOP: features['stop'] = 1
    #rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    #if action == rev: features['reverse'] = 1

    #return features

  #def getWeights(self, gameState, action):
    #return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

##*****************************************************END OF SECONDARY AGENT***************************************************************************************
##*****************************************************Joint Particle Functions*************************************************************************************
##******************************************************************************************************************************************************************

##******************************************************************************************************************************************************************
##******************************************************FEATURE EXTRACTOR BUSINESS**********************************************************************************
##******************************************************************************************************************************************************************

##******************************************************************************************************************************************************************
##******************************************************************************************************************************************************************
##********************************************* POTENTIAL REPRESENTATIONS OF ENEMY AGENTS **************************************************************************

##******************************************************************************************************************************************************************
##******************************************************************************************************************************************************************
##******************************************************************************************************************************************************************
##******************************************************************INFERENCE MODULE BUSINESS***********************************************************************

