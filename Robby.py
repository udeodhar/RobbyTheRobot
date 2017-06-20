########################################################################################
#Title		: Simulation of Robby the Robot.
#Author		: Unmesh Mahendra Deodhar
#Description: 
########################################################################################

import numpy as np
import random
import matplotlib.pyplot as plt 

QMatrix = None

#Hyperparameters...
noofEpisodes = 5000
learningRate = 0.25
gamma = 0.1
steps = 200

#Declaring constants as static variables...
INVALID = -1
WALL = 0
EMPTY = 1
CAN = 2

###################################CLASS CAN GRID##########################################
class Grid:
	def __init__(self, location = (-1, -1)):
		self.locationofRobby = location
		self.grid = np.zeros((10,10))
		for i in range(10):
			for j in range(10):
				self.grid[i,j] = bool(random.getrandbits(1))
		#self.printGrid()
	
	def setLocation(self, (xArg, yArg)):
		self.locationofRobby = (xArg, yArg)
	
	def	printGrid(self):
		print "......................"
		for i in range(10):
			for j in range(10):
				if (i,j) != self.locationofRobby:
					if self.grid[i,j]:
						print "0",
					else:
						print " ",
				else:
					if self.grid[i,j]:				
						print "$",
					else:
						print "R",
			print ""
		print "......................"	
	
	
	def isCan(self, xArg, yArg):
		return self.grid[xArg, yArg]
###################################END OF CLASS CAN GRID##########################################

###################################CLASS ROBBY##########################################
class Robby:

	def __init__(self):
		
		"""
		..........................................................
		States are represented in a sequence
		(Here, North, South, West, East)
		..........................................................
		""" 
		self.locationX = random.randint(0,9)
		self.locationY = random.randint(0,9)
		#print "Location: ",(self.locationX,self.locationY)
		self.grid = Grid((self.locationX, self.locationY))
		
		self.stateHere = INVALID
		self.stateNorth = INVALID
		self.stateSouth = INVALID
		self.stateWest = INVALID
		self.stateEast = INVALID

		self.senseState()

		self.reward = 0
		
		
	def senseState(self):
		###Here state
		self.stateHere = CAN if self.grid.grid[self.locationX, self.locationY] else EMPTY
		
		###West
		if self.locationY == 0:
			self.stateWest = WALL
		elif self.grid.grid[self.locationX, self.locationY-1]:
			self.stateWest = CAN
		else:
			self.stateWest = EMPTY
		
		###East
		if self.locationY == 9:
			self.stateEasst = WALL
		elif self.grid.grid[self.locationX, self.locationY+1]:
			self.stateEast = CAN
		else:
			self.stateEast = EMPTY

		###North
		if self.locationX == 0:
			self.stateNorth = WALL
		elif self.grid.grid[self.locationX-1, self.locationY]:
			self.stateNorth = CAN
		else:
			self.stateNorth = EMPTY
		
		###South
		if self.locationX == 9:
			self.stateSouth = WALL
		elif self.grid.grid[self.locationX+1, self.locationY]:
			self.stateSouth = CAN
		else:
			self.stateSouth = EMPTY
			
		
	def setLocation(self, xArg, yArg):
		self.locationX = xArg
		self.locationY = yArg
		self.grid.setLocation((xArg, yArg))
		
	def getLocation(self):
		return (self.locationX, self.locationY)
		
	def getState(self):
		#print "self.stateHere: ",self.stateHere
		#print "self.stateNorth: ", self.stateNorth
		#print "self.stateSouth: ", self.stateSouth
		#print "self.stateWest: ", self.stateWest
		#print "self.stateEast: ", self.stateEast
		return ((3**4)*(self.stateHere) + (3**3)*(self.stateNorth) + (3**2)*(self.stateSouth) + (3**1)*(self.stateWest) + (3**0)*self.stateEast )
	
	def moveNorth(self):
		if self.locationX == 0:
			self.reward -= 5
			self.senseState()
			#self.grid.printGrid()
			return -5
		else:
			self.locationX -= 1
			self.grid.setLocation((self.locationX, self.locationY))
			self.senseState()
			#self.grid.printGrid()
			return 0
			
	def moveSouth(self):
		if self.locationX == 9:
			self.reward -= 5
			self.senseState()
			#self.grid.printGrid()
			return -5
		else:
			self.locationX += 1
			self.grid.setLocation((self.locationX, self.locationY))
			self.senseState()
			#self.grid.printGrid()
			return 0
			
	def moveWest(self):
		if self.locationY == 0:
			self.reward -= 5
			self.senseState()
			#self.grid.printGrid()
			return -5
		else:
			self.locationY -= 1
			self.grid.setLocation((self.locationX, self.locationY))
			self.senseState()
			#self.grid.printGrid()
			return 0
			
	def moveEast(self):
		if self.locationY == 9:
			self.reward -= 5
			self.senseState()
			#self.grid.printGrid()
			return -5
		else:
			self.locationY += 1
			self.grid.setLocation((self.locationX, self.locationY))
			self.senseState()
			#self.grid.printGrid()		
			return 0
			
	def pickCan(self):
		if self.grid.grid[self.locationX, self.locationY]:
			self.reward += 10
			self.grid.grid[self.locationX, self.locationY] = False
			self.senseState()
			#self.grid.printGrid()
			return 10
		else:
			self.reward -= 1
			#self.grid.printGrid()
			return -1
		
		
	def takeAction(self, rand):
		if rand == 0:
			return self.pickCan()
		elif rand == 1:
			return self.moveNorth()
		elif rand == 2:
			return self.moveSouth()
		elif rand == 3:
			return self.moveWest()
		elif rand == 4:
			return self.moveEast()
	"""
	States are represented in a sequence
	(Here, North, South, West, East)	
	"""
	def setState(self, tuple):
		(self.stateHere, self.stateNorth, self.stateSouth, self.stateWest, self.stateEast) = tuple
	
###################################END OF CLASS ROBBY##########################################
	
###################################MAIN##########################################
def main():
	QMatrix = np.zeros((3**5, 5))
	cumulativeReward = 0
	rewards = []
	epsilon = 1		#Reduce it by 0.01 after every 50 epochs, minimum is 0.1
	for episode in range(noofEpisodes):
		robby = Robby()
	#	robby.grid.printGrid()
		for step in range(steps):
			reward = 0
			currentState = robby.getState()
			#print "currentState: ",currentState
			action = None
			if epsilon > (1 - epsilon):
				action = random.randint(0,4)
				reward = robby.takeAction(action)
			#	print "action: ",action
			else:
				if np.count_nonzero(QMatrix[robby.getState(),:]) == 0:
					action = random.randint(0,4)
				else:
					action = np.argmax( QMatrix[robby.getState(),:] )
				#print action
				reward = robby.takeAction( action )
			QMatrix[currentState, action] += learningRate*( reward + gamma*( max( QMatrix[robby.getState(), :] ) ) - QMatrix[currentState, action] )
		#if (episode%50 == 0) and epsilon >= 0.11:
	#		epsilon -= 0.01
		if episode%100 == 0:
			rewards.append(robby.reward)

###########################################
	epsilon = 0
	testingRewards = []
	for episode in range(noofEpisodes):
		robby = Robby()
	#	robby.grid.printGrid()
		for step in range(steps):
			reward = 0
			currentState = robby.getState()
			#print "currentState: ",currentState
			action = None
			if np.count_nonzero(QMatrix[robby.getState(),:]) == 0:
				action = random.randint(0,4)
			else:
				action = np.argmax( QMatrix[robby.getState(),:] )
				#print action
			reward = robby.takeAction( action )
#			QMatrix[currentState, action] += learningRate*( reward + gamma*( max( QMatrix[robby.getState(), :] ) ) - QMatrix[currentState, action] )
		#if (episode%50 == 0) and epsilon >= 0.11:
		#	epsilon -= 0.01
		if episode%100 == 0:
			testingRewards.append(robby.reward)


			
		"""
		for i in range(3**5):
			for j in range(5):
				print QMatrix[i,j],
			print ""
		"""
		#print "Episode: ",episode,"		Reward: ",robby.reward	
	print "Average: ",np.average(testingRewards)
	print "Standard Deviation: ",np.std(testingRewards)
	
	plt.xlabel('\nEpisodes\n', size=16)
	plt.ylabel('\nSum of Rewards\n', size=16)
	#plt.tick_params(axis='x', pad=30)
	#plt.xticks(range(0,noofEpisodes,100),va="bottom",ha="left")
	plt.plot(rewards)
#	plt.xticks(range(0,noofEpisodes*100))
	plt.savefig('Result.png')
###################################END OF MAIN##########################################

main()