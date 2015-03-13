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

def closestInstance(pos, searchMatrix, walls):
  """
  Finds instance nearest a position, based on 
  a passed in matrix of boolean values, 
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
    # if we find an instance at this location then exit
    if searchMatrix[pos_x][pos_y]:
      return dist
    # otherwise spread out from the location to its neighbours
    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    for nbr_x, nbr_y in nbrs:
      fringe.append((nbr_x, nbr_y, dist+1))
  # no instance found
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
    #self.distancer = Distancer( getLayout('defaultCapture') )
    #self.distancer.getMazeDistances()

  def getFeatures(self, gameState, action):
    features = util.Counter()
    features['score'] = self.agent.getScore(gameState)
    
    # ***************** features of agents ***********************
    initialPos = gameState.getInitialAgentPosition(self.agent.index)
    agentState = gameState.getAgentState(self.agent.index)
    scaredTime = agentState.scaredTimer
    # instead of generating successor, find next position given action
    # NOTE: assumes legal action given
    if action not in gameState.getLegalActions(self.agent.index): 
      print "WHAT ARE YOU DOING HERE, YOUNG MAN?"
    x, y = agentState.getPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    myPos = next_x, next_y
    features['no-stop'] = -10 if action is Directions.STOP else 1
    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1 if not agentState.isPacman else 2
    # Computes whether our teammate is on defense (1) or offense (0)
    features['friendOnD'] = 2 if not gameState.getAgentState(self.agent.friendIndex).isPacman else 1
    friendPos = gameState.getAgentPosition(self.agent.friendIndex)
    # enemyPositions refers to the best guess from exact inference module
    self.enemyPositions = self.agent.getEnemyPositions(gameState)
    enemies = []
    for enemyIndex in self.agent.enemyIndices:
      enemy.index = enemyIndex
      enemy.pos = self.enemyPositions[enemyIndex]
      enemy.state = gameState.getAgentState(enemyIndex)
      enemies.append(enemy)
    
    # ************************* game features **************************************
    walls = gameState.getWalls()
    " LIKE A BAT OUTTA HELL "
    features['initial-column'] = -666 if myPos[0] is initialPos[0] else 1337
    " WHAT'RE MY OPTIONS, HMMMM??? "
    features['available-moves-from-successor'] = len(Actions.getLegalNeighbors(myPos, walls))
    eatFood = self.agent.getFood(gameState)
    defendFood = self.agent.getFoodYouAreDefending(gameState)
    "TIME TO PLAY SOME D"
    if features['onDefense'] == 1 and scaredTime == 0:
    # if we're a ghost and not under the effect of a power capsule
      try:
        defendCapsules = self.agent.getCapsulesYouAreDefending(gameState)
        capsuleThreat = min(enemies, key = lambda enemy: closestInstance(enemy.pos, defendCapsules, walls))
        enemyToCapsule = closestInstance(capsuleThreat, defendCapsules, walls)
        features['threatened-capsule'] = float(5/enemyToCapsule)
        features['distance-to-capsule-threat'] = float(4/self.agent.getDistance(myPos, capsuleThreat.pos))
      except IndexError:
        "No capsules to defend"
      # tuple of enemy instance closest to food and positions - (distance, enemy position)
      foodThreat = min(enemies, key = lambda enemy: closestInstance(enemy[0], defendFood, walls))
      enemyToFood = closestInstance(foodThread, defendFood, walls)
      features['threatened-food'] = float(2/enemyToFood)
      features['distance-to-food-threat'] = float(3/self.agent.getDistance(myPos, foodThreat.pos))
    elif features['onDefense'] == 1 and scaredTime > 0:
    # do something for scared ghost
      print 'aaaah! IMMA SCARED GHOST!'
    else:
    # offensive manuevers
      try:
        eatCapsules = util.matrixAsList(self.agent.getCapsules(gameState))
        closestCapsule = min(eatCapsules, key = lambda capsule: self.agent.getDistance(myPos, capsule))
        closestEnemy = min(enemies, key = lambda enemy: closestInstance(enemy.pos, myPos, walls))
        enemyDistance = self.agent.getDistance(myPos, closestEnemy.pos)
        enemyToCapsule = self.agent.getDistance(closestCapsule, closestEnemy.pos)
        meToCapsule = self.agent.getDistance(myPos, closestCapsule)
        if meToCapsule < enemyToCapsule:
          features['capsule-craving'] = float(5/enemyDistance*meToCapsule)
      except IndexError:
        "No capsules to eat"
        
    # trend towards middle
    features['halfway-point-distance'] = 1 / (  abs( next_x - (gameState.getWalls().width / 2) )  )
    # and away from the column of initial spawning
    features['distance-from-home'] = 10 * abs(initialPos[0] - myPos[0])
    # and trend away from teammate
    features['spread-and-destroy'] = 2 * self.distancer.getDistance(myPos, friendPos)
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