

### Traffic congestion

Traffic congestion, also known as traffic jam, is a complex problem with no 
definitive answer. Increased economic prosperity leads to a considerable 
growth in urban traffic demands. This over usage of road infrastructures 
coupled with stochastic events often leads to traffic jams inducing
delays for the users and other negative impacts such as wasting time, 
wasting fuel, increased air pollution, increased vehicles maintenance, 
human health problems, or emergency service degradation, to name a few.

> __Definition:__
> Network congestion is the number of actors passing through a point
> in a defined time window.

According to the largest study on global traffic from [INRIX](http://inrix.com/scorecard/),
, the traffic congestion cost is 305 billion $ in 2017 in U.S. 

![Traffic congestion economic loss](fig/economic_loss.png)

These statistics suggest a correlation between economic loss and
 number of drivers. Interestingly, the longest congestion time
correlates with city size and thus road network size.

If we think about it, traffic networks are similar to natural networks, such 
as biological or ecological networks.

![interactom](fig/640px-Network_of_how_100_of_the_528_genes_identified_with_significant_differential_expression_relate_to_DISC1_and_its_core_interactors.png)

These networks are rarely congested thanks to regulatory mechanism that 
auto-adapt traffic demands. Could we design regulatory mechanisms that 
can handle road traffic as efficiently as natural networks ?

### Reinforcement Learning

> __Definition:__
> Reinforcement learning is a process that map situation to actions.

In reinforcement learning, the learner must discover this mapping by
trial and error and evaluation with a delayed reward.

![RL](fig/RL.png)

Given a state _s_ resuming the observations from an environment, 
an agent must decide which action _a_ to take. At first, the rules
of the environment are not known, thus the agent will take random 
actions. Following action, the environment will return a signal
to the agent, called the reward _r_, to evaluate its success in 
accomplishing the task. The reward signal is then utilized by the 
agent to update its policy so that the next time it encounter the
same state it has a higher probability of choosing the right
action to get a success.

Recent advances in Deep Learning had a huge impact on Reinforcement
Learning by broadening the state space that the agent can ingest
to predict actions. For example, deep nets such as convolutional
networks allows agent to observe the environment directly from
pictures.

### AI controlled traffic

From the literature, there is no satisfactory theoretical model
of network traffic and its congestion. It is thus tempting to 
develop some empirical predictive models that could help us
regulate traffic flows with the goal of limiting traffic congestion.

Current research is focused on traffic light control at intersection.
In this work, we would like to consider the traffic network (or sub-network)
as whole for the reinforcement learning problem. Here, we are more 
interested in optimizing the congestion level of the whole network, 
than optimizing travel time for the users of the network. Indeed,
 it is likely that optimizing one traffic light at an intersection 
 could cause traffic congestion at another intersection.The rational
behind this approach is that traffic network congestion is dependent
on both network architecture and usage.  We anticipate that a RL agent 
will make the best use of the whole network usage compared to informations
 at intersections only. 
 
### Simulation

#### Network modelling

According to the definition of traffic congestion, we care about 
the **number of actors** going through a **road** during a 
**time** period.

The unit is the road. Roads can have different sizes and contains
different number of actors. We choose to represent the roads as nodes
and intersections as edges to easily extract roads informations. Nodes
have a single index while edges have 2 indices (node1,node2).

We defined a class for the generation of random graph with fixed number of
entry, exit and core nodes. The creation of edges are randomized and 
parameterized with the number of paths and the path depths.

![rand_graph](/fig/random_graph.png)

Core nodes have a max capacity visualized by the size of the node.

