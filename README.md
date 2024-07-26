# MAC-Aggregation-Optimization

In this repository, we optimize the dependency between the message and the tag to increase the throughput and reliability of message authentication in lossy channels.

We have implemented the dependency graph, reward, and metric calculations in the src/Auth.py file.

Moreover we optimize this graph using Gurobi, implemented in the src/tagModel.py file.

Lastly, some more examples, such as framework playground, strength number, and run the optimizer model, are in the example folder.

To run all the codes, you need the pulp and the groubi optimizer


