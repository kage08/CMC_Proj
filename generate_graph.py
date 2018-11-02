import numpy as np

def create_complete_graph(num_vertices):
    '''
    Input
    -----
        num_vertices : (int) Number of vertices in the graph
    Output
    ------
        Returns the weighted complete graph as a numpy matrix
    '''
    graph = np.random.random_integers(0,100,(num_vertices,num_vertices))
    graph = (graph + graph.T)
    return graph
