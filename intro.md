

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

