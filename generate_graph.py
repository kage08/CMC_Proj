import numpy as np
from itertools import *


def create_complete_graph(num_vertices):
    '''
    Input
    -----
        num_vertices : (int) Number of vertices in the graph
    Output
    ------
        Returns the weighted complete graph as a numpy matrix
    '''
    graph = np.random.random_integers(0,100,(num_vertices,num_vertices))
    graph = (graph + graph.T)
    np.fill_diagonal(graph,0)
    return graph


def GeneratePaths(BF_arrMatrix):
    # Extracting the nodes of the TSP
    lstNodes = [node for node in range(len(BF_arrMatrix))]
    # Remove the last city to generate non cyclic permutations
    last_node = lstNodes.pop()
    # Enumerating all the paths from the nodes
    lstPermutations = list(permutations(lstNodes))
    # Constructing a tree
    lstTree = list(map(list, lstPermutations))
    # Closing the paths / Constructing full cycles
    for path in lstTree:
        path.append(last_node)
        path.append(path[0])
    return lstNodes, lstTree

def BruteForce(BF_arrMatrix):
    '''
    Function implementing the brute-force solution to TSP
    Input
    -----
        BF_arrMatrix : (matrix) Complete weighted graph
    Output
    ------
        Optimal solution path
    '''
    # Start time
    # Generate all the possible paths
    lstNodes, lstTree = GeneratePaths(BF_arrMatrix)
    # Calculating the cost of each cycle
    lstCostList = []
    for cycle in lstTree:
        # Initialize cost for each cycle
        numCostPerCycle = 0
        # Convert each 2 nodes in a cycle to an index in the input array
        for index in range(0,(len(lstNodes)-1)):
            # CostPerCycle is calculated from the input Matrix between
            #   each 2 nodes in a cycle
            numCostPerCycle = numCostPerCycle + BF_arrMatrix[cycle[index]][cycle[index+1]]
        lstCostList.append(numCostPerCycle)
    # Calculating the least cost cycle
    numLeastCost = min(lstCostList)
    numLeastCostIndex = lstCostList.index(numLeastCost)
    BF_output = ["Brute Force", numLeastCost, lstTree[numLeastCostIndex]]
    return(BF_output)

def BranchNBound(BnB_arrMatrix):
    '''
    Function implementing the branch and bound solution to TSP
    Input
    -----
        BnB_arrMatrix : (matrix) Complete weighted graph
    Output
    ------
        Optimal solution path
    '''
    # Generate the TSP nodes and all the possible paths
    lstNodes, lstTree = GeneratePaths(BnB_arrMatrix)
    # Calculating the cost of each cycle
    lstCostList = []
    # Initialize the current best/optimal cost to infinity
    numCurrentBestCost = float("inf")
    for cycle in lstTree:
        # Initialize cost for each cycle
        numCostPerCycle = 0
        # Convert each 2 nodes in a cycle to an index in the input array
        for index in range(0,(len(lstNodes)-1)):
            # CostPerCycle is calculated from the input Matrix between
            #   each 2 nodes in a cycle
            numCostPerCycle = numCostPerCycle + BnB_arrMatrix[cycle[index]][cycle[index+1]]
            # Check the current accumlated cost against the Current Best Cost
            if (numCostPerCycle >= numCurrentBestCost):
                numCostPerCycle = float("inf")
                break
        # Add the first cycle cost as the best one
        if (numCurrentBestCost == float("inf")):
            numCurrentBestCost = numCostPerCycle
        # if a better cost is found, update the numCurrentBestCost variable
        elif (numCostPerCycle < numCurrentBestCost):
            numCurrentBestCost = numCostPerCycle
        # Add the current cycle cost to the cost list
        lstCostList.append(numCostPerCycle)
    # Calculating the least cost cycle
    numLeastCost = min(lstCostList)
    numLeastCostIndex = lstCostList.index(numLeastCost)
    BnB_output = ["Branch and Bound", numLeastCost, lstTree[numLeastCostIndex]]
    return(BnB_output)

def greedy(cities, start_city=0):
    #Path of the tour
    path = [start_city]
    #Cost of the tour
    cost = 0
    #Assume the first city as starting point, create a temporary value
    temp = start_city
    #Hold the position in the list
    position = start_city
    #The number of the unvisited cities
    flag = len(cities) - 1
    #Current cost
    current_cost = 0
    while flag > 0:
        #Find the nearest city
        for x in range(0, len(cities)):
            #x not in path, mean we don't care about the cities that we visited
            if (cities[temp][x] != 0) and (x not in path):
                if current_cost == 0:
                    current_cost = cities[temp][x]
                    position = x
                if current_cost > cities[temp][x]:
                    current_cost = cities[temp][x]
                    position = x
        cost += int(current_cost)
        #Reset current cost for next calculating
        current_cost = 0
        temp = position
        path.append(position)
        if flag == 1:
            #Add the connected path from last city to the start city
            current_cost = cities[position][start_city]
            cost += current_cost
            path.append(start_city)
        flag -= 1
    algorithm = "Greedy"
    result = [algorithm, cost, path]
    # print "The cost of the tour is:"+str(result[1])
    # print "The path of the tour is:"+str(result[2])
    # print "The time to finish is:"+str(result[3])+" in second"
    return result


def better_greedy(cities):
    '''
    Function implementing the greedy solution to TSP
    Input
    -----
        cities : (matrix) Complete weighted graph
    Output
    ------
        Optimal solution path
    '''
    # print "greedy algorithm is running. Please wait!"
    result = greedy(cities, 0)
    result_temp = []
    i = 0
    while i < len(cities):
        result_temp = greedy(cities, i)
        if result[1] > result_temp[1]:
            result = result_temp
        i += 1
    # print "The best result:"
    # print "The cost of the tour is:"+str(result[1])
    # print "The path of the tour is:"+str(result[2])
    # print "The time to finish is:"+str(result[3])+" in second"
    # print result
    return result

'''
Example Usage
-------------
    >>> g = create_complete_graph(30)
    >>> print(BruteForce(g))
    >>> print(BranchNBound(g))
    >>> print(better_greedy(g))
    >>> #for higher number of vertices use only greedy approach
'''
