#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import copy
import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from queue import PriorityQueue
import random


class TSPSolver:
    def __init__(self, gui_view):
        self.worstCost = 1
        # FIXME This should find & keep track of the actual worst cost thus far.
        # The fitness function relies on having this number!
        self._scenario = None
        self.bssf = None

    def setupWithScenario(self, scenario: Scenario):
        """
        This sets up the TSPSolver with a given scenario.
        :param scenario:
        :return:
        """
        self._scenario = scenario

    def defaultRandomTour(self, time_allowance=60.0):
        """
        This finds a tour through the cities randomly.
        :param time_allowance: The amount of time allotted to run for.
        :return: The results of the randomly found tour.
        """
        results = {}
        cities = self._scenario.cities
        ncities = len(cities)
        foundTour = False
        count = 0
        self.bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = [cities[perm[i]] for i in range(ncities)]
            # Now build the route using the random permutation
            # for i in range(ncities):
            # 	route.append(cities[perm[i]])
            self.bssf = TSPSolution(route)
            count += 1
            if self.bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = self.bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = self.bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def greedy(self, time_allowance=60.0):
        """
        This finds the greedy solution for a starting tour.
        :param time_allowance: The time allotted to run for.
        :return: The results of the found greedy solution.
        """

        count = 0
        self.bssf = None
        start_time = time.time()
        # This runs the greedy algorithm starting at each city and keeps the best one
        for city in self._scenario.cities:
            solution = self.get_greedy_route(city)
            if solution.cost != np.inf:
                count += 1
                if self.bssf is None or self.bssf.cost > solution.cost:
                    self.bssf = solution

        end_time = time.time()
        return {'cost': self.bssf.cost,
                'time': end_time - start_time,
                'count': count,
                'soln': self.bssf,
                'max': None,
                'total': None,
                'pruned': None}

    def get_greedy_route(self, current_city):
        """
        This calculates a greedy route from a specific starting city.
        :param current_city: The city to start from.
        :return: The TSPSolution of the greedy route.
        """
        visited_cities = []
        # This loops through adding the closest city onto the route
        while True:
            visited_cities.append(current_city)
            try:
                # This finds the next city by getting the closest city that hasn't been already visited
                next_city = min(((city, current_city.costTo(self._scenario.cities[city.get_index()]))
                                 for city in self._scenario.cities
                                 if city not in visited_cities), key=lambda e: e[1])[0]
            # There is an error here when there is no next city to visit. This is how we know it has visited
            #   all of the cities.
            except ValueError:
                return TSPSolution(visited_cities)
            current_city = next_city

    def branchAndBound(self, time_allowance=60.0):
        """
        This uses the branch and bound algorithm to efficiently cut out longer paths to efficiently look
        for the shortest path.
        :param time_allowance: The allotted time to look.
        :return: The best found path when either the best path is found or time runs out.
        """

        count = 0
        added_nodes = 0
        pruned_nodes = 0
        max_nodes = 0
        start_time = time.time()
        # We start with a bssf found by the greedy algorithm.
        self.greedy(time_allowance)
        priority = PriorityQueue()
        # I create the matrix by looping through each city and getting the distance to each other city.
        matrix = np.array([[i.costTo(self._scenario.cities[j.get_index()])
                            for j in self._scenario.cities]
                           for i in self._scenario.cities])
        # For this, we start at the first city and put it in the queue
        node = Node(matrix, 0, [0])
        priority.put(node)
        added_nodes += 1
        # While there is still time and the queue isn't empty, we keep searching.
        while time.time() < (start_time + time_allowance) and priority.qsize() > 0:
            # This keeps track of how big the queue gets.
            if priority.qsize() > max_nodes:
                max_nodes = priority.qsize()

            # This gets a node from the queue. If it's lower bound is still under the bssf, then we check
            #   to see if it has reached all of the cities.
            node = priority.get()
            if node.lower_bound < self.bssf.cost:
                if node.test():
                    # If it has reached all of the cities, we check to make sure it can loop back to the
                    #   starting city, then check and see if it is a better solution than the bssf. If it
                    #   is, it becomes the bssf and we keep searching.
                    solution = TSPSolution(node.solution_cities(self._scenario))
                    if solution.cost < self.bssf.cost:
                        count += 1
                        self.bssf = solution
                else:
                    # If it hasn't reached all of the cities, we create new nodes from each possible next
                    #   cities. If the lower_bound is still better than the bssf, then we add it to the queue.
                    for i in range(len(matrix)):
                        if i not in node.visited_cities:
                            new_node = node.next_city(i)
                            if new_node.lower_bound < self.bssf.cost:
                                priority.put(new_node)
                                added_nodes += 1
                            else:
                                pruned_nodes += 1
            else:
                pruned_nodes += 1

        end_time = time.time()
        return {'cost': self.bssf.cost if self.bssf is not None else np.inf,
                'time': end_time - start_time,
                'count': count,
                'soln': self.bssf,
                'max': max_nodes,
                'total': added_nodes,
                'pruned': pruned_nodes}

    def fancy(self, time_allowance=60.0):
        start_time = time.time()
        """
        <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution,
        time spent to find best solution, total number of solutions found during search, the
        best solution found.  You may use the other three field however you like.
        algorithm</returns>
        """
        """
        GA(Fitness())
            self.population = InitializePopulation()
            while not done do:
                parents = Select(population, Fitness())
                children = Crossover(parents)
                children = Mutate(children)
                population = Survive(population, children, Fitness())
            return HighestFitness(population)
        """
        # Tuning Variables:
        num_parents = 8  # the number of parents that will be selected to repopulate
        num_children = 4  # the number of children each parent will have
        self.min_population_size = 50
        self.max_population_size = 500
        self.percentageToKill = 0.10
        max_jump = None  # the furthest away the mutate function will swap cities within a route/candidate

        self.greedy(time_allowance)
        population = [self.bssf]
        worstCost = self.bssf.cost
        for i in range(0, self.min_population_size - 1):
            newSolution = self.defaultRandomTour()['soln']
            if newSolution.cost > worstCost:
                worstCost = newSolution.cost
            population.append(newSolution)
            #Starts us off with a population of at least our minimum size
        done = False
        while not done and time.time() < (start_time + time_allowance):
            parents = self.select(population, num_parents)
            children = self.crossover(parents, num_children)
            children = self.mutate(children)
            population.extend(children)
            self.survival_function(population)

        end_time = time.time()
        self.bssf =  self.highestFitness(population)
        return {'cost': self.bssf.cost,
                'time': end_time - start_time,
                'count': 420,
                'soln': self.bssf,
                'max': None,
                'total': None,
                'pruned': None}

    def highestFitness(self, population):
        result = population[0]
        for candidate in population:
            if self.fitness_function(candidate) > self.fitness_function(result):
                result = candidate
        return result

    def select(self, candidates: List[TSPSolution], num_parents=2):
        #create a list of tuples with [solution : fitnessValue]
        candidates.sort(key=self.fitness_function)
        return candidates[:num_parents]

    def crossover(self, parents: List[TSPSolution], num_children):
        children = []
        for i in range(num_children):
            mother = random.choice(parents)
            father = random.choice(parents)
            child_cities = []

            # pick a random sequence from the father and add to the child
            father_start = random.randrange(len(father.route))
            for i in range(father_start, father_start + random.randint(1, len(father.route))):
                child_cities.append(father.route[(i % len(father.route))])

            # fill in what's missing from the mother in the order those cities appear
            i = random.randrange(len(mother.route))
            while len(child_cities) < len(mother.route):
                if mother.route[i] not in child_cities:
                    child_cities.append(mother.route[i])
                i += 1
            
            children.append(TSPSolution(child_cities))

        return children

    def mutate(self, children: List[TSPSolution], max_jump=None):
        '''
        Slightly change each child route given in hopes that the 'mutation' will lead
        to greater survivability. It does it by swapping two random elements in the route.
        :param children: A list of TSPSolutions that will be mutated one by one
        :param max_jump: The maximum distance between two swapped elements.
        :return: The mutated list of TSPSolutions.
        '''
        for child in children:
            ncities = len(child.route)
            if max_jump is None or max_jump > ncities - 1:
                max_jump = ncities - 1
            swap1 = random.randint(0, ncities-1)
            min = swap1 - max_jump if swap1 - max_jump >= 0 else 0
            max = swap1 + max_jump if swap1 + max_jump < ncities else ncities - 1
            swap2 = random.randint(min, max)
            child.route[swap1], child.route[swap2] = child.route[swap2], child.route[swap1]
            if child.cost > worstCost:
                worstCost = child.cost

        return children

    def survival_function(self, population):
        # This function doesn't have input or output- it operates on the current population, and does it in-place.
        # This function does NOT guarantee a constant population size. It would be pretty easy to guarantee that,
        # so if we want to we can.
        numIndividualsToRemove = math.trunc(len(population) * self.percentageToKill)
        if (len(population) - numIndividualsToRemove) > self.max_population_size:
            numIndividualsToRemove = len(population - self.max_population_size)
        for i in range(0, numIndividualsToRemove):
            if len(population) > self.min_population_size:
               didKill = False
               while (didKill == False):
                    currentIndex = math.trunc((random.random() * 10000000000) % len(population))
                    currentIndividual = population[currentIndex]
                    fitnessScore = self.fitness_function(currentIndividual)
                    randVal = random.random() * 100
                    if (randVal < fitnessScore):
                       del population[currentIndex]
                       didKill = True

        # TODO This function.... well, we don't really know how long it takes. Because it grabs
        # random members of the population to see if they'll survive to the next round, it could theoretically
        # run forever if it continuously fails to remove the selected population member.
        # On the other hand, if it removes a population member on the first try each iteration, it could run in
        # o(n) time (technically O(n * percentageToKill) time, but that's approximated to O(n) time)

    def fitness_function(self, solution: TSPSolution) -> float:
        # This function returns a decimal between 0 and 1 as a fitness score.
        # It first calculates what percentage of the worst path length encountered so far
        # the current path takes up, then squares that number to give the more fit (lower-cost)
        # individuals an even better chance at surviving.
        fractionalValue = solution.cost / self.worstCost
        return fractionalValue ** 2
        # TODO This function runs in constant time- the length of the input bears no sway on how long it takes.
        # It also doesn't take up any additional space- it works in-place.


class Node:

    def __init__(self, matrix: np.array, lower_bound: int, visited_cities: List[int]):
        self.matrix = matrix
        self.lower_bound = lower_bound
        self.visited_cities = visited_cities
        self.reduce_cost()

    def next_city(self, next_city):
        """
        This creates a new node from the current node by giving it the next city to visit.
        :param next_city: The next city to visit.
        :return: A new node that is a copy of the current node except having visited the given city.
        """

        matrix = np.array(self.matrix)
        visited_cities = copy.copy(self.visited_cities)
        last_city = visited_cities[-1]
        distance = matrix[last_city, next_city]
        lower_bound = self.lower_bound + distance
        matrix[last_city, :] = np.inf
        matrix[:, next_city] = np.inf
        matrix[next_city, last_city] = np.inf
        visited_cities.append(next_city)

        return Node(matrix, lower_bound, visited_cities)

    def reduce_cost(self):
        """
        This reduces the cost of paths in a matrix while increasing the lower bound of the path.
        :return: None
        """

        # This subtracts the minimum from each row while adding it to the lower bound
        for row in self.matrix:
            minimum = np.min(row)
            if minimum == np.inf:
                continue
            row -= minimum
            self.lower_bound += minimum

        # This subtracts the minumum from each column while adding it to the lower bound.
        for col in self.matrix.transpose():
            minimum = np.min(col)
            if minimum == np.inf:
                continue
            col -= minimum
            self.lower_bound += minimum

    def solution_cities(self, scenario):
        """
        The creates a list of cities from the info in the given scenario.
        :param scenario: The scenario containing all of the information.
        :return: The list of cities as a path from the current node and given scenario.
        """
        return [scenario.cities[i] for i in self.visited_cities]

    def test(self):
        """
        This checks to see if this node has visited all of the cities.
        :return: True if it has visited all of the cities, else false.
        """
        return len(self.matrix) == len(self.visited_cities)

    def __cmp__(self, other):
        """
        This is how two different nodes are compared. The one with the lowest priority is the one
        that is the smallest.
        :param other: The other node to compare this node with.
        :return: 1 if greater, 0 if equal, and -1 if smaller.
        """
        if self.get_priority() > other.get_priority():
            return 1
        elif self.get_priority() == other.get_priority():
            return 0
        else:
            return -1

    def __eq__(self, other):
        return self.__cmp__(other) == 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __le__(self, other):
        return self.__cmp__(other) <= 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __ge__(self, other):
        return self.__cmp__(other) >= 0

    def get_priority(self):
        """
        This is how the priority for a node is calculated.
        :return: The priority of this node.
        """

        # The priority is calculated by first dividing the lower bound by number of edges in the current path.
        #   This gives us an estimate of the average edge length for this path. We then add that distance to the
        #   lower bound for each edge left to be calculated in the path. This gives us an estimate of how long the
        #   path will be.
        num_visited = len(self.visited_cities)
        num_cities = len(self.matrix)
        edges_left = num_cities - num_visited + 1
        if num_visited != 0:
            avg_edge = self.lower_bound / num_visited
        else:
            return self.lower_bound
        return self.lower_bound + (edges_left * avg_edge)
