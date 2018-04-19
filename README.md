## ML graph problems

### Graph navigation

* given an historical collection of graph paths, 
how would you predict the next move in the current path?
    * _use case: predict node congestion_
    
    * experiment: 
        1. define a random weighted directed graph 
        with random node queue capacity
        2. select entry nodes and exit nodes (degree 1)
        3. define a random number of actors at entry nodes at each time step
        4. define a shortest path for each actors at entry node 
        from the list of exit nodes. 
        5. move actors queues according to edge speed (nb/step) 
        and next nodes capacity
        6. Pop out head node from the path list of each actors when moving.
        7. record the number of actors at each nodes for each time steps
        
    * analysis: 
        1. which nodes are congested (nb actors == max capacity)?
        2. which nodes are frequently congested ?
        3. what is the relation between connectivity and congestion ?
        4. visualize historical loading for each nodes
        5. visualize correlation between nodes loading
        
    * prediction:
        1. without knowing actors paths, predict the probability of 
        congestion for each nodes at the next step.
        2. choosing a start and end node, predict the best next node according to 
        the probability of nodes congestion. What is the travel time difference
        between path choosen by historical loading and probability of congestion?
        3. predict each actors' number of steps to their destination
    
### Graph recommendation

* given a network of actors and targets, 
how would you define groups of targets ?
    * _use case: recommendations_

### Graph planning

* given concurrent execution graphs, historical execution timing and 
limited resource quantity, how to predict running and finishing jobs execution
for the next time steps ?
    * _use case: graph monitoring, dynamic planning_