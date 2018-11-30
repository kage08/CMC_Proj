# CMC_Proj: Hierarchical Framework for Genetic Algorithms
## Authors: [Harshavardhan P K](https://github.com/CS15B061) and Kousik Krishnan

###Introduction
We implemented a hierarchical framework for genetic algorithms where we serch through space of objective functions and then search solution for these objectives.

### Requirements
1. Numpy
2. Tensorflow(optional)
3. Matplotlib

### Scripts
1. __run.py__: Visualizes objective function for regression
2. __run1.py__: Runs experiments for metasolver for soft-TSP (Travelling Salesman)
3. __run2.py__: Runs experiments for metasolver for polynomial regression

### Important Files
1. __GAtspSolver.py__: Implementation for TSP GA
2. __MetaTSPsolver.py__: Implements metasolver for soft TSP
3. __Polysolver.py__: Implements gradient based and GA solvers for polynomial regression
4. __MetaRegressorSolver.py__: Implements metasolver for polynomial regression over multiple objective functions.