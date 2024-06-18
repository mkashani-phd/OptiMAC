# MAC-Aggregation-Optimization

In this repository we optimzie the dependency between the message and the tag to increase the throughput and reliability of message authentication in lossy channel.

We have implemented the dependency graph, reward and some more metric calucalation in the src/Auth.py file.

More over we optimize this graph using Gurobi, implemented in the src/tagModel.py file.

Lastly some more example such as frame work playground, strength number, and run the optimizer model are in the example folder.

To run all the codes you need the pulp and the groubi optimizer


