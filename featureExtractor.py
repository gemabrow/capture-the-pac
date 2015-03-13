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

NO_DBZ = 0.0000000000001 # No division by zero -- not to confused with those against the DragonBall Z series

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
      enemy = {}
      enemy['index'] = enemyIndex
      enemy['pos'] = self.enemyPositions[enemyIndex]
      enemy['isPacman'] = gameState.getAgentState(enemyIndex).isPacman
      enemy['scaredTimer'] = gameState.getAgentState(enemyIndex).scaredTimer
      enemies.append(enemy)
    
    # ************************* game features **************************************
    walls = gameState.getWalls()
    
    " LIKE A BAT OUTTA HELL "
    features['initial-column'] = -666 if myPos[0] is initialPos[0] else 1337
    
    " WHAT'RE MY OPTIONS, HMMMM??? "
    features['available-moves-from-successor'] = len(Actions.getLegalNeighbors(myPos, walls))
    # trend towards middle
    features['halfway-point-distance'] = 1 / (  abs( next_x - (walls.width / 2) )  )
    # and away from the column of initial spawning
    features['distance-from-home'] = 10 * abs(initialPos[0] - myPos[0])
    # and trend away from teammate
    features['spread-and-destroy'] = 2 * self.agent.distancer.getDistance(myPos, friendPos)
        
    "TIME TO PLAY SOME D"
    # if we're a ghost and not under the effect of a power capsule
    invaders = [enemy for enemy in enemies if enemy['isPacman']]
    if features['onDefense'] == 1 and invaders:
      defendFood = self.agent.getFoodYouAreDefending(gameState)
      try:
        defendCapsules = self.agent.getCapsulesYouAreDefending(gameState)
        capsuleThreat = min(invaders, key = lambda enemy: closestInstance(enemy['pos'], defendCapsules, walls))
        enemyToCapsule = closestInstance(capsuleThreat['pos'], defendCapsules, walls)
        features['threatened-capsule'] = float( 5/(NO_DBZ+enemyToCapsule) )
        features['distance-to-capsule-threat'] = float( 4/(NO_DBZ+self.agent.distancer.getDistance(myPos, capsuleThreat['pos'])) )
      except (IndexError, ValueError):
        pass
      # tuple of enemy instance closest to food and positions - (distance, enemy position)
      foodThreat = min(invaders, key = lambda enemy: closestInstance(enemy['pos'], defendFood, walls))
      enemyToFood = closestInstance(foodThreat['pos'], defendFood, walls)
      features['threatened-food'] = float( 2/(NO_DBZ+enemyToFood) )
      features['distance-to-food-threat'] = float( 3/(NO_DBZ+self.agent.distancer.getDistance(myPos, foodThreat['pos'])) )
    # Our ghost is scared!
    elif features['onDefense'] == 1 and scaredTime > 1:
      distanceToCenter = min(abs(legalPos[0] - walls.width/2) for legalPos in Actions.getLegalNeighbors(myPos, walls))
      features['FLEE'] = distanceToCenter
    else: # offensive manuevers
      ghosts = [enemy for enemy in enemies if not enemy['isPacman']]
      food = self.agent.getFood(gameState)
      eatFood = food.asList()
      closestEnemy = min(ghosts, key = lambda enemy: self.agent.distancer.getDistance(enemy['pos'], myPos))
      enemyDistance = self.agent.distancer.getDistance(myPos, closestEnemy['pos'])
      closestFood = min(eatFood, key = lambda food: self.agent.distancer.getDistance(food, myPos))
      meToFood = self.agent.distancer.getDistance(closestFood, myPos)
      if closestEnemy['isPacman'] and closestEnemy['scaredTimer'] >= 2:
        # TIME TO RAGE
        features['RAGE-RAGE-RAGE'] = 666/( NO_DBZ + (min(enemyDistance, meToFood)) )
      else:
        # PLAY IT SMART, KID
        ghostsInLegalNeighbors = sum(myPos in Actions.getLegalNeighbors(g['pos'], walls) for g in ghosts)
        features["#-of-ghosts-1-step-away"] = ghostsInLegalNeighbors
        enemyToFood = self.agent.distancer.getDistance(closestFood, closestEnemy['pos'])
        features['gotta-eat'] = float(enemyDistance/meToFood)
        if enemyToFood > meToFood and enemyDistance > 2:
          features['gotta-eat'] *= 3
        try:
          eatCapsules = util.matrixAsList(self.agent.getCapsules(gameState))
          closestCapsule = min(eatCapsules, key = lambda capsule: self.agent.distancer.getDistance(myPos, capsule))
          enemyToCapsule = self.agent.distancer.getDistance(closestCapsule, closestEnemy['pos'])
          meToCapsule = self.agent.distancer.getDistance(myPos, closestCapsule)
          if meToCapsule < enemyToCapsule:
            features['capsule-craving'] = float( 5/(NO_DBZ + (enemyDistance*meToCapsule) ) )
        except (IndexError, ValueError):
          pass
      
    return features