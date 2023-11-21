from player import Player
import numpy as np
import copy
from numpy import random
import random

class Evolution():

    def __init__(self):
        self.game_mode = "Neuroevolution"

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        mu = 0

        child.nn.weights[0] += np.random.normal(mu, 1, child.nn.weights[0].shape)
        child.nn.weights[1] += np.random.normal(mu, 1, child.nn.weights[1].shape)

        child.nn.biases[0] += np.random.normal(mu, 0.8, child.nn.biases[0].shape)
        child.nn.biases[1] += np.random.normal(mu, 0.8, child.nn.biases[1].shape)

        return child

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.game_mode) for _ in range(num_players)]

        else:

            fitnesses = []
            for player in prev_players:
                fitnesses.append(player.fitness)
            new_players = []
            new_playersM = []
            parents = random.choices(prev_players, weights=fitnesses, cum_weights=None, k=num_players)
            for player in parents:
                new_players.append(self.clone_player(player))
                new_playersM.append(self.mutate(self.clone_player(player)))

            newPlayers = []
            for i in range(0, len(new_players), 2):
                firstPlayer, secondPlayer = self.crossOver(new_players[i], new_players[i + 1], 0.4)
                newPlayers.append(self.mutate(firstPlayer))
                newPlayers.append(self.mutate(secondPlayer))
            return newPlayers
            # return new_playersM

    def crossOver(self, firstPlayer, SecondPlayer, Probability):
        if random.random() < Probability:
            firstPlayer.nn.weights[0], SecondPlayer.nn.weights[0] = self.matrixCrossOver(firstPlayer.nn.weights[0],
                                                                                         SecondPlayer.nn.weights[0])
            firstPlayer.nn.weights[1], SecondPlayer.nn.weights[1] = self.matrixCrossOver(firstPlayer.nn.weights[1],
                                                                                         SecondPlayer.nn.weights[1])
            firstPlayer.nn.biases[0], SecondPlayer.nn.biases[0] = self.matrixCrossOver(firstPlayer.nn.biases[0],
                                                                                       SecondPlayer.nn.biases[0])
            firstPlayer.nn.biases[1], SecondPlayer.nn.biases[1] = self.matrixCrossOver(firstPlayer.nn.biases[1],
                                                                                       SecondPlayer.nn.biases[1])

        return firstPlayer, SecondPlayer

    def matrixCrossOver(self, firstMatrix, secondMatrix):
        x = random.randint(1, firstMatrix.size)
        shape = firstMatrix.shape
        flatten1 = firstMatrix.flatten()
        flatten2 = secondMatrix.flatten()
        tmp = flatten2[:x].copy()
        flatten2[:x], flatten1[:x] = flatten1[:x], tmp
        newMatrix1 = flatten1.reshape(shape)
        newMatrix2 = flatten2.reshape(shape)

        return newMatrix1, newMatrix2

    def next_population_selection(self, players, num_players):

        players.sort(key=lambda x: x.fitness, reverse=True)

        fitnesses = []
        ProbFitness = []
        totalFitness = 0
        for player in players:
            fitnesses.append(player.fitness)
        for i in fitnesses:
            totalFitness += i
        for i in fitnesses:
            ProbFitness.append(i / totalFitness)
        newPlayers = random.choices(players, weights=ProbFitness, cum_weights=None, k=num_players)

        # average = 0
        # min = players[len(players) - 1].fitness
        # max = players[0].fitness
        # for i in players:
        #     average += i.fitness
        # average /= len(players)
        # f = open("data.txt", "a")
        # f.write(str(min) + "\n")
        # f.write(str(max) + "\n")
        # f.write(str(average) + "\n")
        return newPlayers
        # return players[:num_players]

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player