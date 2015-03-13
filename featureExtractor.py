# Gerald Brown (gemabrow@ucsc.edu) & Alfred Young (ayoung4@ucsc.edu) // CMPS 140 -- Winter 2015
# featureExtractors.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from distanceCalculator import Distancer
from game import Directions, Actions
from layout import getLayout
import util

def closestFood(pos, food, walls):
  """
  Finds food nearest a position, 
  based on a passed in matrix of food, 
  and returns the maze distance to it
  """
  initialPos = pos
  fringe = [(pos[0], pos[1], 0)]
  expanded = set()
  while fringe:
    pos_x, pos_y, dist = fringe.pop(0)
    if (pos_x, pos_y) in expanded:
      continue
    expanded.add((pos_x, pos_y))
    # if we find a food at this location then exit
    if food[pos_x][pos_y]:
      return dist
    # otherwise spread out from the location to its neighbours
    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    for nbr_x, nbr_y in nbrs:
      fringe.append((nbr_x, nbr_y, dist+1))
  # no food found
  return None

class MasterExtractor:
  # NOTE: gameState.getAgentState(index) = gameState.data.agentStates[index] <--- could hold exact position of opposing team
  # OR use starting position plus AgentState.getDirection & vector difference to figure new position
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
    self.distancer = Distancer( getLayout('defaultCapture') )
    self.distancer.getMazeDistances()

  def getFeatures(self, gameState, action):
    features = util.Counter()
    features['score'] = self.agent.getScore(gameState)
    
    # ***************** agent features ***********************
    initialPos = gameState.getInitialAgentPosition(self.agent.index)
    agentState = gameState.getAgentState(self.agent.index)
    scaredTime = agentState.scaredTimer
    features['no-stop'] = -10 if action is Directions.STOP else 1
    # Computes whether we're on the red or blue team
    features['red'] = 1 if self.agent.red else -1
    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1 if not agentState.isPacman else 0
    # Computes whether our teammate is on defense (1) or offense (0)
    features['friendOnD'] = 1 if not gameState.getAgentState(self.agent.friendIndex).isPacman else 0
    friendPos = gameState.getAgentPosition(self.agent.friendIndex)
    # instead of generating successor, find next position given action
    # NOTE: assumes legal action given
    x, y = agentState.getPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    myPos = next_x, next_y
    # ************************* game features **************************************
    walls = gameState.getWalls()
    features['available-moves'] = len(Actions.getLegalNeighbors(myPos, walls))
    eatFood = self.agent.getFood(gameState)
    defendFood = self.agent.getFoodYouAreDefending(gameState)
    try:
      eatCapsules = util.matrixAsList(self.agent.getCapsules(gameState))
      if len(eatCapsules) > 0:
        minDistance = 1/(min([self.distancer.getDistance(myPos, capsule) for capsule in eatCapsules])+0.01)
        features['distanceToEatCapsule'] = minDistance
    except IndexError:
      "Index error for capsules thrown"
    self.enemyPositions = self.agent.getEnemyPositions(gameState)
    enemy1, enemy2 = [self.enemyPositions[enemyIndex] for enemyIndex in self.agent.enemyIndices]
    features['initial-column'] = -1000 if myPos[0] is initialPos[0] else 5
    features['distance-to-enemy1'] = self.distancer.getDistance(myPos, enemy1)
    features['distance-to-enemy2'] = self.distancer.getDistance(myPos, enemy2)
    features['halfway-point-distance'] = abs(next_x - (gameState.getWalls().width / 2) )
    features['distance-from-home'] = 10 * abs(initialPos[0] - myPos[0])
    # Compute distance to the nearest food
    foodList = eatFood.asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.distancer.getDistance(myPos, food) for food in foodList])
      features['distanceToEatFood'] = minDistance
    if features['friendOnD'] == 1:
      features['distanceToEatFood'] /= 100
      features['distance-from-home'] **= 2
    foodList = defendFood.asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.distancer.getDistance(myPos, food) for food in foodList])
      features['distanceToDefendFood'] = 1/(minDistance+0.01)
    
    # count the number of ghosts 1-step away
    ghosts = [self.enemyPositions[enemyIndex] for enemyIndex in self.agent.enemyIndices
             if not gameState.getAgentState(enemyIndex).isPacman]
    if ghosts:
      ghostScaredTimer = min (gameState.getAgentState(enemyIndex).scaredTimer for enemyIndex in self.agent.enemyIndices)
      features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
      # if there is no danger of ghosts then add the food feature
      if features["#-of-ghosts-1-step-away"] == 0 and eatFood[next_x][next_y]:
        features["eats-food"] = 10
      elif ghostScaredTimer > 1:
        features["sorry-not-scared"] = min(self.agent.distancer.getDistance(myPos, ghost) 
                                           for i, ghost in enumerate(ghosts)
                                           if gameState.getAgentState(self.agent.enemyIndices[i]).scaredTimer > 1)
    
    invaders = [self.enemyPositions[enemyIndex] for enemyIndex in self.agent.enemyIndices
                if gameState.getAgentState(enemyIndex).isPacman and not None]
    
    if invaders and features['onDefense'] == 1:
      if scaredTime == 0:
        features['activeDefense'] = min(self.agent.distancer.getDistance(invader, myPos) for invader in invaders)
      else:
        features['fleefullyWatching'] = -1 * scaredTime * features['halfway-point-distance']
    return features