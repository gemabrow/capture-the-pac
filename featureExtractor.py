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
import time, util, math

NO_DBZ = 1 # No division by zero -- not to confused with those against the DragonBall Z series

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

def asList(self, key = True):
    list = []
    for x in range(self.width):
      for y in range(self.height):
        if self[x][y] == key: list.append( (x,y) )
    return list
  
    
def matrixDiff(matrix_prev, matrix_curr):
  l_p = matrix_prev.asList()
  l_c = matrix_curr.asList()
  return len(l_p) - len(l_c)

def _cellIndexToPosition(grid, index):
  x = index / grid.height
  y = index % grid.height
  return x, y

def _positionToCellIndex(grid, pos):
  x, y = pos
  index = x * grid.height + y % grid.height
  return index

class IdentityExtractor:
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[(state,action)] = 1.0
    return feats

class SimpleExtractor():
  """
  Returns simple features for a basic reflex Pacman:
  - whether food will be eaten
  - how far away the next food is
  - whether a ghost collision is imminent
  - whether a ghost is one step away
  """
  def __init__(self, myAgent):
    # passing in agent should give us access to beliefs
    self.agent = myAgent
    
  def getFeatures(self, state, action):
    # extract the grid of food and wall locations and get the ghost locations
    food = self.agent.getFood(state)
    walls = state.getWalls()
    enemyPositions = self.agent.getEnemyPositions(state)
    myPos = state.getAgentPosition(self.agent.index)
    friendPos = state.getAgentPosition(self.agent.friendIndex)
    features = util.Counter()
    
    features["bias"] = 1.0

    features["distance-to-teammate"] = float(self.agent.distancer.getDistance(myPos, friendPos)) / (walls.width * walls.height)
    # compute the location of pacman after he takes the action
    x, y = state.getAgentPosition(self.agent.index)
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    
    # count the number of enemies 1-step away
    closeGhosts = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in enemyPositions.values())
    if state.getAgentState(self.agent.index).isPacman is True:
      features["#-of-ghosts-1-step-away"] = closeGhosts
    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 100.0

    dist = closestInstance((next_x, next_y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = -1 * dist
    features.normalize()
    return features
  
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