# Gerald Brown (gemabrow@ucsc.edu) & Alfred Young (ayoung4@ucsc.edu) // CMPS 140 -- Winter 2015
# featureExtractors.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import Directions, Actions
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
    
  def getFeatures(self, gameState, action):
    features = util.Counter()    
    features['bias'] = 1.0
    
    # ***************** agent features ***********************
    # Computes whether we're on the red or blue team
    features['red'] = 1 if self.agent.red else 0
    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1 if self.agent.isPacman else 0
    # Computes whether our teammate is on defense (1) or offense (0)
    features['friendOnD'] = 1 if self.agent.friendIndex

    # instead of generating successor, find next position given action
    # NOTE: assumes legal action given
    x, y = gameState.getAgentState(self.agent.index).getPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    
    # ************************* game features **************************************
    eatFood = self.agent.getFood(gameState)
    defendFood = self.agent.getFoodYouAreDefending(gameState)
    eatCapsules = self.agent.getCapsules(gameState)
    defendCapsules = self.agent.getCapsulesYouAreDefending(gameState)
    walls = gameState.getWalls()
    distribution = self.agent.getDistribution(gameState)
    possiblePositions = self.agent.getPositions(distribution)
    # creates a list of each enemy's actual position or, 
    # if not available, their most likely position
    enemiesPositions = [gameState.getAgentPosition(enemy) for enemy in self.agent.enemyAgents 
                        if gameState.getAgentPosition(enemy) is not None 
                        else possiblePositions[enemy]]
    # Compute distance to the nearest food
    foodList = food.asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([agent.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToEatFood'] = minDistance
    foodMatrix.asList() = agent.getFoodYouAreDefending(gameState)
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([agent.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToDefendFood'] = minDistance
    
    enemies = [successor.getAgentState(agent.index) for enemy in agent.enemyAgents]
    enemyPos = []
    for enemyState in enemies:
      enemyPos.append(enemyState.getPosition() if enemyState.getPosition() is not None else 
    features['nearest-enemy'] = min(successor.getAgentPosition(enemyIndex) for enemyIndex in agent.enemyAgents)
    return features
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    # extract the grid of food and wall locations and get the ghost locations
    food = state.getFood()
    walls = state.getWalls()
    ghosts = state.getGhostPositions()


    
    features["bias"] = 1.0
    
    # compute the location of pacman after he takes the action
    x, y = state.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)
    
    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0
    
    dist = closestFood((next_x, next_y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = float(dist) / (walls.width * walls.height) 
    features.divideAll(10.0)
    return features
