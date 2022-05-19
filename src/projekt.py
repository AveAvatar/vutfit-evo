#!/usr/bin/env python3
# python_version = '3.9'
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Tadeáš Kachyňa, <xkachy00@stud.fit.vutbr.cz>
# Created Date: 08/05/2022
# ---------------------------------------------------------------------------
""" Implementation of Graph Coloring Problem by Genetic Algoritm""" 
# ---------------------------------------------------------------------------
from array import *
from ast import arguments
from termcolor import colored
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import argparse

# -------------------------- Argument Parsing -------------------------------

parser = argparse.ArgumentParser(description="EVO 2022 / Graph Coloring Problem")
parser.add_argument("--size", action="store", dest="size")
parser.add_argument("--colors", action="store", dest="colors")
parser.add_argument("--graph", action="store", dest="graph")
parser.add_argument("--file", action="store", dest="file")
arguments = parser.parse_args()

# ---------------------------------------------------------------------------

if arguments.file == 'yes':
	sys.stdout = open('file.txt', 'w')
g = open("stats.txt",'w')

Gen = np.array([])
Fit = np.array([])
all_gen = np.array([])
all_fitness = np.array([])

'''Create Graph'''
# graph is represented by a matrix with values of ones and zeros
# ones are representing vertices which are connected together, zeros do not
# variable "n" represents the size of that matrix
# can be modified via user's arguments
graph = [] 
n = 400

if arguments.size != None:
	try:
		n = int(arguments.size)
	except:
		print("ERROR > Argument [size] is not valid.")
		exit(1)

for i in range(n):
	vertex = []
	for j in range(n):
		vertex.append(random.randint(0, 1))
	graph.append(vertex)
for i in range(n):
	for j in range(0, i):
		graph[i][j] = graph[j][i]
for i in range(n):
	graph[i][i] = 0
for vector in graph:
	print(vector)

'''Upper Bound for Coloring'''
# algoritm starts with 'upper bound' of colors and decreases it each time when 
# solution with zero fitness is found
# this number can be set via user's arguments
max_num_colors = 1

for i in range(n):
	if sum(graph[i]) > max_num_colors:
		max_num_colors = sum(graph[i]) + 1

if arguments.colors != None:
	try:
		max_num_colors= int(arguments.colors)
	except:
		print("ERROR > Argument [colors] is not valid.")
		exit(1)

print("\nStarting number of colors is: " + str(max_num_colors), end="\n\n")
number_of_colors = max_num_colors

# function 'print_individual' prints individual. Based on their fitness function, it chooses the color
def print_individual(best_fitness, max_num_colors, fittest_individual, gen):
	print("Generation: ", colored(gen, 'blue'), " | Best Fitness: ", end="")
	if best_fitness > max_num_colors / 2: 
		print(colored(str(best_fitness), 'red'), end="")
	elif best_fitness >= max_num_colors / 3: 
		print(colored(str(best_fitness), 'yellow'), end="")
	else: 
		print(colored(str(best_fitness), 'green'), end="")
	print(" | Individual: ", fittest_individual)



'''Genetic Algorithm'''
fittest_individual = None
condition = True
start = time.time()
while(condition and number_of_colors > 0):
	def create_individual():
		individual = []
		for i in range(n):
			individual.append(random.randint(1, number_of_colors))
		return individual

	population_size = 20
	generation = 0
	population = []
	# creating the population of individuals
	for i in range(population_size):
		individual = create_individual()
		population.append(individual)

	'''Fitness'''
	# To define a fitness function, i assign a penalty of 1 for every edge that has the same colored vertices incident on it.
 	# penalty(i, j) = 1 if there is an edge between i and j
	# =​ 0 Otherwise
	def fitness(graph, individual):
		fitness = 0
		for i in range(n):
			for j in range(i, n):
				if(individual[i] == individual[j] and graph[i][j] == 1):
					fitness += 1
		return fitness

	# fitness function used for advanced crossover, same as the one above
	# but it also returns two vectors with ones and zeros, which say
	# where is the problem 
	def fitnessCross(graph, individual):
		fitness = 0
		vector = [0] * len(individual)
		for i in range(n):
			for j in range(i, n):
				if(individual[i] == individual[j] and graph[i][j] == 1):
					fitness += 1
					vector[i] = 1
					vector[j] = 1
		return fitness, vector

	'''Crossover'''
	# CROSSOVER 1 - advanced - changes only the problematic vertices
	def crossover_only_bad_fitness(parent1, parent2):
		fitness_pos_1, fitness_vec_1 = fitnessCross(graph, parent1)
		fitness_pos_2, fitness_vec_2 = fitnessCross(graph, parent2)		
		child1 = []
		child2 = []	
		probability = 0.8
		check = random.uniform(0, 1)
		if(check <= probability):
			if fitness_pos_1 != 0:
				for i in range(len(parent1)):
					if fitness_vec_1[i] == 1:
						child1.append(parent2[i])
						child2.append(parent1[i])
					else:
						child1.append(parent1[i])
						child2.append(parent2[i])
			elif fitness_pos_2 != 0:
				for i in range(len(parent2)):
					if fitness_vec_2[i] == 1:
						child1.append(parent1[i])
						child2.append(parent2[i])
					else:
						child1.append(parent2[i])
						child2.append(parent1[i])
			if fitness_pos_1 == 0 and fitness_pos_2 == 0:
				child1 = parent1
				child2 = parent2
		else:
			child1 = parent1
			child2 = parent2

		return child1, child2
	
	# CROSSOVER 2 - basic
	# a point on both parents’ vector representation is picked 
	# randomly and designated a ‘crossover point’
	def random_crossover(parent1, parent2):
		position = random.randint(2, n-2)
		child1 = []
		child2 = []
		probability = 0.8
		check = random.uniform(0, 1)
		if(check <= probability):
			for i in range(position+1):
				child1.append(parent1[i])
				child2.append(parent2[i])
			for i in range(position+1, n):
				child1.append(parent2[i])
				child2.append(parent1[i])
		else:
			child1 = parent1
			child2 = parent2

		return child1, child2

	# MUTATION 1
	# there is some chance when it happens.
	# if so, it chooses random vertex and assign it a color
	# based on neighboring vertices
	def mutation(individual, probability):
		check = random.uniform(0, 1)
		if(check <= probability):
			position = random.randint(0, n-1)
			neighbors = np.array(graph[position]) 
			neighbors = np.where(neighbors == 1)[0] # looks for neighbors with the choosen vertex
			used_colors = [] # colors already used in individual 
			all_colors = np.arange(1, number_of_colors+1) # all available colors
			for index in neighbors:
				if individual[index] == individual[position]:
					used_colors.append(individual[index])
			available_colors = set(all_colors) - set(used_colors) # difference set of all and used colors
			available_colors = list(available_colors)
			if len(available_colors) != 0 and len(available_colors) > max_num_colors/2:
				individual[position] = random.choice(available_colors)
			else:
				individual[position] = random.randint(1, number_of_colors)
			

		return individual

	# MUTATION 2 - Mutation based on colors' presence 
	# first i count percentage of each color and its presence in a individual
	# than order them descending, cut the set in half and randomly choose
	# the number
	def mutation_based_on_presence(individual, probability):
		check = random.uniform(0, 1)
		color_presence = {i:0 for i in range(1,number_of_colors+1)}
		for color in range(1, number_of_colors+1): # counting color's presence in the individual
			presence = individual.count(color)
			color_presence[color] = presence / len(individual)
		chosen_colors = dict(sorted(color_presence.items(), key=lambda item: item[1])) # sorting them desceding
		length = round((len(chosen_colors)) / 2) 
		half_chosen_colors = []
		j = 1
		for i in list(chosen_colors): # chosing the first half 
			half_chosen_colors.append(i)
			if j == length:
				break
			j += 1
		
		if(check <= probability):
			position = random.randint(0, n-1)
			individual[position] = random.choice(half_chosen_colors)
		return individual


	# MUTATION 3 - All Colors Mutation
	# the easiest implementation of mutation, again there is some probability
	# when it happens. If so, it chooses random vertex and random color
	# (from all available colors)
	def mutation_all_colors(individual, probability):
		check = random.uniform(0, 1)
		if(check <= probability):
			position = random.randint(0, n-1)
			individual[position] = random.randint(1, number_of_colors)
		return individual


	'''Roulette Wheel Selection'''
	def roulette_wheel_selection(population):
		total_fitness = 0
		for individual in population:
			total_fitness += 1/(1+fitness(graph, individual))
		cumulative_fitness = []
		cumulative_fitness_sum = 0
		for i in range(len(population)):
			cumulative_fitness_sum += 1 / \
				(1+fitness(graph, population[i]))/total_fitness
			cumulative_fitness.append(cumulative_fitness_sum)

		new_population = []
		for i in range(len(population)):
			roulette = random.uniform(0, 1)
			for j in range(len(population)):
				if (roulette <= cumulative_fitness[j]):
					new_population.append(population[j])
					break
		return new_population

	best_fitness = fitness(graph, population[0])
	last_best = fittest_individual
	fittest_individual = population[0] 
	gen = 0
	# this loop runs for "x" generations or untill the fitness function is equal to zero
	# it applies selection and genetic operators - mutation and crossover
	# then it counts each individual's fitness and print them
	while(best_fitness != 0 and gen != 3000): 
		gen += 1
		population = roulette_wheel_selection(population)
		new_population = []
		random.shuffle(population)
		for i in range(0, population_size-1, 2):
			child1, child2 = random_crossover(population[i], population[i+1])
			new_population.append(child1)
			new_population.append(child2)
		for individual in new_population:
			if gen < 500:
				individual = mutation_all_colors(individual, 0.6)
			else:
				individual = mutation_all_colors(individual, 0.3)
		population = new_population
		best_fitness = fitness(graph, population[0])
		fittest_individual = population[0]
		for individual in population:
			if(fitness(graph, individual) < best_fitness):
				best_fitness = fitness(graph, individual)
				fittest_individual = individual
		if gen % 10 == 0:
				print_individual(best_fitness, max_num_colors, fittest_individual, gen)
				
		all_fitness = np.append(all_fitness, best_fitness)
		Gen = np.append(Gen, gen)
		Fit = np.append(Fit, best_fitness)
		

	all_gen = np.append(all_gen, gen)
	
	print("Using ", number_of_colors, " colors : ")
	print("Generation: ", colored(gen, 'blue'), " | Best Fitness: ", end="")
	if best_fitness > max_num_colors / 2: 
		print(colored(str(best_fitness), 'red'), end="")
	elif best_fitness >= max_num_colors / 3: 
		print(colored(str(best_fitness), 'yellow'), end="")
	else: 
		print(colored(str(best_fitness), 'green'), end="")

	print(" | Individual: ", fittest_individual)
	print("\n\n")
	if(best_fitness != 0):
		condition = False		
		print("Graph is ", number_of_colors+1, " colorable")
		end = time.time()

	else:
		Gen = np.append(Gen, gen)
		Fit = np.append(Fit, best_fitness)
		all_fitness = np.append(all_fitness, best_fitness)
		plt.plot(Gen, Fit)
		plt.xlabel("generation")
		plt.ylabel("best-fitness")
		if arguments.graph == 'yes':
				plt.show()			
		Gen = []
		Fit = []
		number_of_colors -= 1

######## Statistics #########

all_gen.sort()
print("Average Fitness: " + str(sum(all_fitness)/len(all_fitness)))
print("Median: " + str(np.median(all_fitness)))
print("Maximum Fitness: " + str(np.max(all_fitness)))
print("Largest generation until the best fitness is found: " + str(all_gen[-2]))
print("Sum of all generations: " + str(np.sum(all_gen)))
if arguments.file == 'yes':
	sys.stdout.close()