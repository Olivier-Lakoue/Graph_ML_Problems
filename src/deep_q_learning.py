import numpy as np

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
        self.shape = (length, 2*states_len + actions_len + 1)
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
