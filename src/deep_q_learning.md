### ToDo: class Memory
A fixed size numpy array storing and recalling past
   * states
   * actions
   * next-states
   * reward
   
It includes a function generating batches.

### ToDo: class ActionSpace
Generating all combinations of actions and producing one_hot vectors
of these combinations. Functions to include:
1. sample: get a random combination.
2. get_nodes: from an action vector, get the list of nodes to inactivate
           in the graph.

### ToDo: class DQN
1. Generate a deep learning model taking as inputs:  
 * states
 * actions
 
and as output:  
 * q_values (expected rewards)
 
2. A fit function generating target q_values and fitting the model
3. A train function with hyperparameters:
 * epsilon : proportion of random actions vs. predicted actions
 * memory size
 * steps
 * gamma : discount factor for past rewards
 * batch size
 
4. An epsilon-greedy scheduler function
5. A report function plotting the rolling mean of the rewards and save on disk


### Tests
1. use deep recurrent keras layers as in `[16]_recurrent_neural_nets.ipynb`
and states sequence preparation as in `[15]_sequence_preparation.ipynb`

2. 
