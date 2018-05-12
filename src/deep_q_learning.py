import numpy as np
from itertools import combinations
from random import randint
from keras.models import Model, load_model
from keras.layers import Dense, Input, multiply
import os
import time
from glob import glob
import pickle

class Memory():
    """
    A ring buffer implementation in numpy.
    """
    def __init__(self, length, states_len, actions_len):
        """
        Memory setup numpay array of shape (length, 2*states_len + actions_len + 1)
        :param length:
        :param states_len:
        :param actions_len:
        """
        self.shape = (length, (2*states_len + actions_len + 1))
        self.sample_shape = tuple(np.subtract(self.shape, (1,0)))
        # an array of random normal dist mu=0.5 std=0.25
        self.memory = 0.25 * np.random.randn(*self.shape) + 0.5
        # clip values to [0,1]
        self.memory = np.where(self.memory > 1., np.ones_like(self.memory), self.memory)
        self.memory = np.where(self.memory < 0., np.zeros_like(self.memory), self.memory)
        # pointer
        self.memory[0] = np.nan
        # bounds
        self.states_length = states_len
        self.actions_length = actions_len

    def load(self, state, actions, next_state, reward):
        """
        Store state, actions, next_state and reward arrays into the circular buffer memory
        :param state:
        :param actions:
        :param next_state:
        :param reward:
        :return:
        """
        # concatenate values
        values = np.hstack((
            state.ravel(),
            next_state.ravel(),
            np.array([reward]),
            actions.ravel()
        ))
        # pointer position
        current_position = np.where(np.isnan(self.memory))[0][0]
        # insert values
        self.memory[current_position, :] = values
        # move pointer
        if current_position == self.memory.shape[0]-1:
            current_position = 0
        else:
            current_position += 1
        self.memory[current_position, :] = np.nan

    def replay(self, batch_size):
        """
        Return a batch of states, actions, next_states and rewards from the memory
        :param batch:
        :return:
        """
        sample_idx = np.random.choice(np.arange(self.sample_shape[0]), size=batch_size)
        samples = self.memory[np.where(~ np.isnan(self.memory))]
        samples = samples.reshape(self.sample_shape)[sample_idx]
        # split batch into states, next_states, reward, actions
        states = samples[:, :self.states_length]
        next_states = samples[:, self.states_length: self.states_length*2]
        rewards = samples[:, self.states_length*2: self.states_length*2+1]
        actions = samples[:, self.states_length*2+1:]
        return states, actions, next_states, rewards

class ActionSpace():
    """
    One hot encoding of actions combinations.
    """
    def __init__(self, rand_graph):
        """
        Get the core_nodes from the submited instance of RandGraph() to
        produce all the combinations of core_nodes. Encode the combination
        vector as one_hot vector.
        :param rand_graph: RandGraph() instance
        """
        self.nodes = rand_graph.core_nodes
        self.combinations = self.action_comb()

    def action_comb(self):
        c = []
        for i in range(1, len(self.nodes)+1):
            c.extend(list(combinations(self.nodes, i)))
        return c

    def sample(self):
        """
        Return a random one_hot vector
        :return: numpy array
        """
        a = np.zeros((1, len(self.combinations)))
        idx = self.combinations[randint(0, len(self.combinations)-1)]
        a[:, self.combinations.index(idx)] = 1.0
        return a

    def get_nodes(self, action):
        """
        Return combination of nodes from supplied action one_hot vector.
        :param action: numpy array
        :return: list of int
        """
        idx = np.where(action == 1.)[1][0]
        return list(self.combinations[idx])

    def get_action(self, model, epsilon, state):
        """
        Get memory based or predicted action vector according to epsilon proba.
        :param model:
        :param epsilon:
        :param state:
        :return:
        """
        if np.random.randn() < epsilon:
            action = self.sample()
        else:
            a = self.sample()
            q_values = model.predict([state, np.ones_like(a)])
            idx = np.argmax(q_values)
            action = np.zeros((1, len(self.combinations)))
            action[:, idx] = 1.
        return action

class DQN():
    """
    Deep Q_learning model an accessory functions.
    """
    def __init__(self, rand_graph, gamma=.9, batch_size=32, mem_size=10000):
        """
        Initial model
        :param length_states: int
        :param length_actions: int
        :param gamma: discount factor for the bellman equation
        :param mem_size: length of the memory to sample from during fitting

        """
        self.gamma = gamma
        self.batch_size = batch_size
        self.rand_graph = rand_graph
        self.action_space = ActionSpace(rand_graph)
        action = self.action_space.sample()
        self.length_actions = action.shape[1]
        self.length_states = len(rand_graph.core_nodes)
        self.model = self.create_model(self.length_states, self.length_actions)
        self.memory = Memory(mem_size, self.length_states, self.length_actions)
        self.total_rewards = []


    def create_model(self, nb_values, nb_actions):
        values_input = Input((nb_values,), name='values')
        action_input = Input((nb_actions,), name='mask')
        x = Dense(32, activation='relu')(values_input)
        x = Dense(32, activation='relu')(x)
        output = Dense(nb_actions)(x)
        filtered_output = multiply([output, action_input])
        model = Model(inputs=[values_input, action_input], outputs=filtered_output)
        model.compile(optimizer='rmsprop', loss='mse')
        return model

    def get_epsilon_for(self, iteration):
        """
        Epsilon-greedy scheduler function
        :param iteration: step
        :return: epsilon(float)
        """
        e = 0.99 - 0.0001 * iteration
        if e < 0.1:
            e = 0.1
        return e

    def fit(self, states, actions, next_states, rewards):
        # get all the Q_values for this specific states
        next_Q_values = self.model.predict([next_states, np.ones_like(actions)])
        # bellman
        Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)[:, None]
        # target Q_values
        out = actions * Q_values
        # train
        self.model.fit([states, actions], out, batch_size=self.batch_size, verbose=0)

    def train(self, steps):
        self.save(new_folder=True)
        # initial state
        state = np.zeros((1,self.length_states))
        for step in range(steps):
            # select an action
            epsilon = self.get_epsilon_for(step)
            action = self.action_space.get_action(self.model, epsilon, state)
            # Act on the graph
            n = self.action_space.get_nodes(action)
            next_state, reward = self.rand_graph.action(n)

            # remember
            self.memory.load(state, action, next_state, reward)
            # store reward
            self.total_rewards.append(reward)

            # get a batch and fit
            states, actions, next_states, rewards = self.memory.replay(self.batch_size)
            self.fit(states, actions, next_states, rewards)

            # prepare for the next step
            state = next_state
        self.save()

    def save(self, new_folder=False):
        if new_folder:
            date = time.strftime("%Y_%m_%d_%H_%M")
            os.mkdir('../models/' + date)
            path = '../models/' + date + '/'
            self.model.save(path + 'model.h5')
            with open(path + 'actionspace.pkl', 'wb') as f:
                pickle.dump(self.action_space, f)
        else:
            try:
                latest = glob('../models/*/')[-1]
                self.model.save(latest + 'model.h5')
                with open(latest + 'actionspace.pkl', 'wb') as f:
                    pickle.dump(self.action_space, f)
            except IndexError:
                print('''Model not saved.\nYou need to create a new folder using the parameter:\nnew_folder=True.''')

    def load(self):
        try:
            latest = glob('../models/*/')[-1]
            self.model = load_model(latest + 'model.h5')
            with open(latest + 'actionspace.pkl', 'rb') as f:
                self.action_space = pickle.load(f)
        except IndexError:
            print('Model not found.')

    def predict(self, state):
        action = self.action_space.sample()
        q_val = self.model.predict([state, np.ones_like(action)])
        idx = np.argmax(q_val)
        act_vect = np.zeros_like(action)
        act_vect[:, idx] = 1.
        return self.action_space.get_nodes(act_vect)

