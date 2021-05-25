from math import sqrt
from PIL import Image
import numpy as np

import os
import json
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

assert __name__ != '__main__', 'Module startup error.'

np.seterr(over='ignore')

# path to MNIST dataset folder (should be configured)
dataset = r'D:\App\Datasets\MNIST'

TRAIN = fr'{dataset}\train'
VALIDATION = fr'{dataset}\validation'
TEST = fr'{dataset}\test'


def preprocessing():
    centralized_data = np.mean([(np.asarray(Image.open(fr'{TRAIN}\{file}')) / 256).tolist()
                                for file in os.listdir(TRAIN)], axis=0)
    np.save('mean', centralized_data)


def get_target(file: str):
    return int(file[-5])


class ReLU:
    @staticmethod
    def calculate(x):
        return x * (x > 0)

    @staticmethod
    def derivative(x):
        return 1. * (x > 0)


class Sigmoid:
    @staticmethod
    def calculate(x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.calculate(x) * (1 - self.calculate(x))


class Network:
    def __init__(self):
        self.__syn0 = None
        self.__syn1 = None
        self.__syn2 = None

        self._activation_function = None
        self.centralized_data = np.load('mean.npy')

    def create_training_structure(self, layout: tuple, function, seed: int = 0):
        hidden1, hidden2 = (size + 1 for size in layout)
        np.random.seed(seed)

        self._activation_function = function

        self.__syn0 = (2 * np.random.sample((785, hidden1)) - 1) / sqrt(785 / 2)
        self.__syn1 = (2 * np.random.sample((hidden1, hidden2)) - 1) / sqrt(hidden1 / 2)
        self.__syn2 = (2 * np.random.sample((hidden2, 10)) - 1) / sqrt(hidden2 / 2)

    def create_test_structure(self, function, seed: int = 0, path: str = 'weights.npz'):
        np.random.seed(seed)
        self._activation_function = function
        self.load(path)

    def save(self, path: str):
        np.savez(path,
                 __syn0=self.__syn0,
                 __syn1=self.__syn1,
                 __syn2=self.__syn2)

    def load(self, path: str):
        weights = np.load(path)
        self.__syn0 = weights['__syn0']
        self.__syn1 = weights['__syn1']
        self.__syn2 = weights['__syn2']

    def to_vector(self, file: str) -> np.ndarray:
        return np.append(np.asarray(Image.open(file)) / 256 - self.centralized_data, 1.).reshape(1, 785)

    def test(self, directory: str) -> float:
        files = np.array(os.listdir(directory))
        return sum(get_target(file) == np.argmax(
            self.forward(self.to_vector(fr'{directory}\{file}'))[-1].ravel()) for file in files) / files.size

    def train(self, b_size: int, mb_size: int, learning_rate: float, annealing=lambda _lr: 0) -> iter:
        assert b_size % mb_size == 0, 'Incorrect batch and mini-batch sizes.'

        batch = np.array(os.listdir(TRAIN))
        np.random.shuffle(batch)
        batch = batch[:b_size].reshape(b_size // mb_size, mb_size)

        for mini_batch in batch:
            delta_syn0, delta_syn1, delta_syn2, accuracy = self._train_mb(mini_batch)
            self.__syn0 += learning_rate * delta_syn0
            self.__syn1 += learning_rate * delta_syn1
            self.__syn2 += learning_rate * delta_syn2
            learning_rate -= annealing(learning_rate)

            yield accuracy

    def _train_mb(self, mini_batch: np.ndarray) -> tuple:
        result = [np.zeros_like(weights) for weights in (self.__syn0, self.__syn1, self.__syn2)]
        goal_cnt = 0

        for file in mini_batch:
            vector = self.to_vector(fr'{TRAIN}\{file}')

            correct_output = get_target(file)
            target = np.zeros(10)
            target[correct_output] = 1.

            layers = self.forward(vector)
            goal_cnt += correct_output == np.argmax(layers[-1].ravel())

            delta_w = self.backward(layers, target)

            for i in range(3):
                result[i] += delta_w[i]

        accuracy = goal_cnt / mini_batch.size
        return *result, accuracy

    def forward(self, input_layer: np.ndarray) -> list:
        layers = [input_layer]
        for weights in (self.__syn0, self.__syn1, self.__syn2):
            layers.append(self._activation_function.calculate(np.dot(layers[-1], weights)))

        return layers

    def backward(self, layers: list, target: np.ndarray) -> list:
        delta_w = [np.zeros_like(weights) for weights in (self.__syn0, self.__syn1, self.__syn2)]
        l0, l1, l2, l3 = layers
        error = l3 - target

        delta3 = error * self._activation_function.derivative(l3)
        delta2 = np.dot(delta3, self.__syn2.T) * self._activation_function.derivative(l2)
        delta1 = np.dot(delta2, self.__syn1.T) * self._activation_function.derivative(l1)

        delta_w[2] -= np.dot(l2.T, delta3)
        delta_w[1] -= np.dot(l1.T, delta2)
        delta_w[0] -= np.dot(l0.T, delta1)

        return delta_w


class Manager:
    def __init__(self, neural_network: Network, directory: str = 'backup sets'):
        self.nn = neural_network
        self.__directory = directory
        self.branch = None

    @staticmethod
    def graph(y: np.ndarray, path: str):
        width = y.size + 1
        x = np.arange(1, width)

        figure = plt.figure()
        axes = plt.axes(xlim=(0, width), ylim=(0, 1))

        figure.set_figwidth(11)
        figure.set_figheight(5)
        figure.subplots_adjust(left=.07, right=.93, top=.93)

        plt.title('Batch statistics', fontsize=15)
        plt.xlabel('mini-batch number', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        axes.grid()
        axes.grid(which='minor')

        axes.yaxis.set_major_locator(ticker.MultipleLocator(.05))
        axes.xaxis.set_major_locator(ticker.MultipleLocator(width // 25))
        axes.xaxis.set_minor_locator(ticker.MultipleLocator(width // 50))

        plt.plot(x, y, color='#3C6C94')
        plt.savefig(fr'{path}\graph.png')
        plt.close()

    def add_branch(self, **kwargs) -> dict:
        self.nn.create_training_structure(**kwargs)

        branches = os.listdir(self.__directory)
        if branches:
            self.branch = fr'{self.__directory}\nn-{max(map(lambda d: int(d[3:]), branches)) + 1:0>6}'
        else:
            self.branch = fr'{self.__directory}\nn-000000'
        os.mkdir(self.branch)

        configs = {
            'layout': kwargs['layout'],
            'activation function': type(kwargs['function']).__name__,
            'seed': kwargs.get('seed') or 0}

        with open(fr'{self.branch}\configs.json', 'w') as file:
            json.dump(configs, file)

        return configs

    def train(self, **kwargs) -> dict:
        start_time = time.time()
        results = np.array([accuracy for accuracy in self.nn.train(**kwargs)])
        total_time = time.time() - start_time

        val_accuracy = self.nn.test(VALIDATION)
        train_accuracy = self.nn.test(TRAIN)

        backups = [el for el in os.listdir(self.branch) if el != 'configs.json']
        if backups:
            backup = fr'{self.branch}\backup-{max(map(lambda d: int(d[7:]), backups)) + 1:0>4}'
        else:
            backup = fr'{self.branch}\backup-0000'
        os.mkdir(backup)

        statistics = {
            'batch size': kwargs['b_size'],
            'mini-batch size': kwargs['mb_size'],
            'learning rate': kwargs['learning_rate'],
            'total time': total_time,
            'validation accuracy': val_accuracy,
            'train accuracy': train_accuracy}

        self.graph(results, backup)
        with open(fr'{backup}\statistics.json', 'w') as file:
            json.dump(statistics, file)
        self.nn.save(fr'{backup}\weights')

        return statistics
