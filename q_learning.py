import random
import numpy as np

from parameters import (
    ALPHA,
    EPSILON,
    GAMMA,
    SEED,
)

from world import EVChargingEnv


class QLearningAgent:

    def __init__(self, seed=SEED):

        self.env = EVChargingEnv(seed)

        # Q-table dimensions:
        # hour (24)
        # battery (11)
        # price (3)
        # action (2)
        self.q = np.zeros((24, 11, 3, 2))

        self.rng = random.Random(seed)

        self.history = []

        # Epsilon decay
        self.epsilon = EPSILON

    # ==============================================================
    # ACTION SELECTION
    # ==============================================================

    def choose(self, state):

        # Explore
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(2), True

        # Exploit
        values = self.q[state]

        best_actions = [
            action
            for action in range(2)
            if values[action] == values.max()
        ]

        return self.rng.choice(best_actions), False

    # ==============================================================
    # ONE TRAINING EPISODE
    # ==============================================================

    def episode(self):

        state = self.env.reset()

        done = False

        total_reward = 0

        while not done:

            action, explored = self.choose(state)

            next_state, reward, done = self.env.step(action)

            old = self.q[state][action]

            if done:
                best_next = 0
            else:
                best_next = self.q[next_state].max()

            new = old + ALPHA * (
                reward
                + GAMMA * best_next
                - old
            )

            self.q[state][action] = new

            total_reward += reward

            state = next_state

        self.history.append(total_reward)

        return total_reward

    # ==============================================================
    # TRAINING
    # ==============================================================

    def train(self, episodes):

        for _ in range(episodes):

            self.episode()

            # ε-decay
            self.epsilon = max(
                0.01,
                self.epsilon * 0.995
            )

    # ==============================================================
    # GREEDY EVALUATION
    # ==============================================================

    def evaluate(self):

        state = self.env.reset()

        total_reward = 0

        done = False

        while not done:

            action = int(
                self.q[state].argmax()
            )

            state, reward, done = self.env.step(action)

            total_reward += reward

        return total_reward