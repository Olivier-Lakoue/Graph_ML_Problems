#### AI controlled traffic networks

Experiments evaluating reinforcement 
learning approaches in traffic flow regulation.
We first need to define a minimal traffic network 
presenting congestion problems.
We will then produce an initial dataset to evaluate if
network congestion is predictable by machine learning.
Finally, reinforcement learning will be tested on a simple
network.
We will discuss about scaling up this experiment 
for real networks.

##### Introduction
__What is traffic congestion ?__
* congestion is the number of actors passing through a point
in a defined time window.
* The main causes are overflows, accidents and road works.

__Why traffic congestion ?__
It is a complex system problem with no definitive solution.
Negative impacts: wasting time, delays, wasted fuel, 
air pollution, vehicles maintenance, human health, emergency 
services, real estate price.
Economic loss: NYC (33.7 M$, 8,6m pop, 1214km2, 91 h/y), 
LA (19.2 M$, 3.98m pop, 1302 km2, 102 h/y),
 SF (10.6M$, 0.884m pop, 600 km2, 79 h/y)
Natural networks, such as biological or ecological networks exists
suggesting that regulation mechanisms discovered in nature could
help us design better traffic regulation.

__What is reinforcement learning ?__
A probabilistic algorithm able to make decisions in a changing
environment. Making decisions to adjust a behaviour in regard to
recent events is a human perk. Bringing computers close to 
human intelligence by replicating this behaviour artificially.

__Why reinforcement learning ?__
There is no definitive theoretical model able to predict and 
regulate traffic flow. We thus need empirical predictive model
to help us design regulatory mechanisms.
Recent advances in deep-learning allowed reinforcement learning 
researchers to obtain spectacular results.


##### Simulation
* __network modeling__

According to the definition of traffic congestion, we care about 
the **number of actors** going through a **road** during a 
**time** period.

* roads will be nodes with a max capacity of actors
* actors will be a simple class defining a network path through
the graph at inception
* time will be model by moving actors to the next node in their path
at each timstep.

* __analyse graph congestion__
    * grid search of parameters 
        * moving actors : 2 - 20
        * paths : 5 - 50
        * entry nodes : 2 - 10 
        * exit nodes : 2 - 10
        * core nodes : 5 - 50

* __seasonal variation of actor loading__
Make the traffic flows more realistic and less predictable by regression.

##### Prediction
* __linear regression__

* __neural net__

* __RNN__


##### Training on a simple network
* __define observation__
* __define actions__
* __define reward__

* __Find base line__

* __define memory__

* __define deep q-learning__

* __training__

* __plotting__


##### Discussion

the importance of reward function in RL is similar to the loss function in SL.