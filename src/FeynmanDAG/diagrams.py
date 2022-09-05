"""

This module part of `FeynmanDAG` and contains definitions on Feynman diagrams and associated matrix elements.

"""
from __future__ import annotations

import matplotlib.pylab as plt
import networkx as nx
import numpy as np

from .interaction import Interactions
from .particles import Particle

__all__ = ["FeynmanDiagrams", "MatrixElement", "ComputeGraph"]


class FeynmanDiagrams:      
    
    """ 
    A class used to obtain the graph whose contain a Feynman Diagram.
    
    ...
    
    
    Attributes
    ----------
    external_particles : list
        A list that contains the external particles of the Feynman Diagram.
    
    operations: list
        A list that contains how the particles will must be combined to obtain the Feynman Diagram.
    
    op_ite: list_iterator
        The iterator that calls a operation when in every generation of the Feynman Diagram.
    
    current_particle_list: list
        The list with the particles that are being combined.
        
    current_operation: list
        The list with the operation that is being realized.
        
    history_of_particle_lists: list
        A list that stores all the generations of particles.
        
    history_of_operations: list
        A list that stores all the results of operations that have been performed.
    
    graph: networkx.classes.graph.Graph
        The graph that stores the Feynman Diagram.

 
    Methods:
    ----------
    number_of_operations():
        Returns the number of operations that must performed to get the Feynman Diagram.
    
    do_next_operation(): 
        It realizes a single operation from the ones that must be performed to get the Feynman Diagram.
    
     get_FD(): 
        Calls recursively the method do_next_operation() until the Feynman Diagram is obtained.
    
    """
    
    def __init__(self,external_particles,operations):
        """
        Constructs a iterator that contains the operations that must be performed and the graph that will store 
        the Feynman Diagram considered.
        
        Parameters
        ----------
            external_particles : list
                A list that contains the external particles of the Feynman Diagram.

            operations: list
                A list that contains how the particles will must be combined to obtain the Feynman Diagram.

            op_ite: list_iterator
                The iterator that calls a operation in every generation of the Feynman Diagram.

            current_particle_list: list
                The list with the particles that are being combined.

            current_operation: list
                The list with the operation that is being realized.

            history_of_particle_lists: list
                A list that stores all the generations of particles.

            history_of_operations: list
                A list that stores all the results of operations that have been performed.

            graph: networkx.classes.graph.Graph
                The graph that stores the Feynman Diagram.    
            
        """
        
        self.external_particles = external_particles
        self.operations = operations
        self.op_ite = iter(self.operations)
        
        self.current_particle_list = external_particles
        self.current_operation = next(self.op_ite)
        
        self.history_of_particle_lists = [self.external_particles]
        self.history_of_operations = []
        
        self.graph = nx.Graph()
    
    @property
    def number_of_operations(self):
        """
        Returns the number of operations that must performed to get the Feynman Diagram.
        
        Returns
        ----------
        int
            The number of operations.
            
        """
        return len(self.operations)
    
    def do_next_operation(self):
        """
        It realizes a single operation from the ones that must be performed to get the Feynman Diagram.
        
        Steps:
            1st. Copying the current list of particles.
            2nd. Poping the particles from the copy according to self.current_operation.keys().
            3rd. Inserting the result of interaction of self.current_operation.keys() into the copy.
            4th. Add the nodes and edges to the graph.
            5th. Overwrite current_particle_list with the resulting list of particles.
            6th. Add the resulting list to the history.
            7th. Add current operation to the history.
            8th. Tick up the operation list to the next operation.
            
        """
        
        operation = self.current_operation
        final_list = self.current_particle_list.copy()
        
        for i in operation:
            final_list.remove(self.current_particle_list[i])
        
        int_part = list(np.array(self.current_particle_list)[list(operation)])
        result = Interactions_FD(int_part)
        final_list.insert(0,result())
        
        part_bf = []
        if len(operation)==2:
            for i in int_part:
                part_bf.append(i.name)
            self.graph.add_nodes_from(part_bf)
            vertex=result().name
            self.graph.add_node(vertex)
            self.graph.add_edges_from([(vertex,part_bf[0]),(vertex,part_bf[1])])
        
        if len(operation)==3:
            for i in int_part:
                part_bf.append(i.name)
            self.graph.add_nodes_from(part_bf)
            vertex=part_bf[1]+"+"+part_bf[2]
            self.graph.add_node(vertex)
            self.graph.add_edges_from([(part_bf[0],vertex),(part_bf[1],vertex),(part_bf[2],vertex)])
            
        if isinstance(result(),Particle):
            self.current_particle_list = final_list
            self.history_of_particle_lists.append(final_list)
        
        self.history_of_operations.append(result())
        
        if len(operation)==2:
            self.current_operation = next(self.op_ite)     
    
    def get_FD(self):
        """
        Calls recursively the method do_next_operation() until the Feynman Diagram is obtained.
        
        Returns
        ----------
        networkx.classes.graph.Graph
            The graph that stores the Feynman Diagram.    
        """
        for n in range(self.number_of_operations):
            self.do_next_operation()
        
        return self.graph

    
class MatrixElement:      
    """ 
    A class used to obtain the matrix corresponding to a Feynman Diagram.
    
    ...
    
    
    Attributes
    ----------
    external_particles : list
        A list that contains the external particles of the Feynman Diagram.
    
    operations: list
        A list that contains how the particles will must be combined to obtain the matrix element.
    
    op_ite: list_iterator
        The iterator that calls a operation in every generation of the Feynman Diagram.
    
    current_particle_list: list
        The list with the particles that are being combined.
        
    current_operation: list
        The list with the operation that is being realized.
        
    history_of_particle_lists: list
        A list that stores all the generations of particles.
        
    history_of_operations: list
        A list that stores all the results of operations that have been performed.

 
    Methods:
    ----------
    number_of_operations():
        Returns the number of operations that must performed to get the Feynman Diagram.
    
    do_next_operation(): 
        It realizes a single operation from the ones that must be performed to get the matrix element.
    
     get_matrix_element(): 
        Calls recursively the method do_next_operation() until the matrix element is obtained.
    
    """
    
    def __init__(self,external_particles,operations):
        """
        Constructs a iterator that contains the operations that must be performed to get the matrix element.
        
        Parameters
        ----------
            external_particles : list
                A list that contains the external particles of the Feynman Diagram.

            operations: list
                A list that contains how the particles will must be combined to obtain the matrix element.

            op_ite: list_iterator
                The iterator that calls a operation in every generation of the matrix element.

            current_particle_list: list
                The list with the particles that are being combined.

            current_operation: list
                The list with the operation that is being realized.

            history_of_particle_lists: list
                A list that stores all the generations of particles.

            history_of_operations: list
                A list that stores all the results of operations that have been performed.
            
        """
        
        self.external_particles = external_particles
        self.operations = operations
        self.op_ite = iter(self.operations)
        
        self.current_particle_list = external_particles
        self.current_operation = next(self.op_ite)
        
        self.history_of_particle_lists = [self.external_particles]
        self.history_of_operations = []
    
    @property
    def number_of_operations(self):
        """
        Returns the number of operations that must performed to get the Feynman Diagram.
        
        Returns
        ----------
        int
            The number of operations.
            
        """
        return len(self.operations)
    
    def do_next_operation(self):
        """
        It realizes a single operation from the ones that must be performed to get the matrix.
        
        Steps:
            1st. Copying the current list of particles.
            2nd. Poping the particles from the copy according to self.current_operation.keys().
            3rd. Inserting the result of interaction of self.current_operation.keys() into the copy.
            4th. Add the nodes and edges to the graph.
            5th. Overwrite current_particle_list with the resulting list of particles.
            6th. Add the resulting list to the history.
            7th. Add current operation to the history.
            8th. Tick up the operation list to the next operation.
            
        """
        operation = self.current_operation
        final_list = self.current_particle_list.copy()
        
        for i in operation:
            final_list.remove(self.current_particle_list[i])
        
        int_part = list(np.array(self.current_particle_list)[list(operation)])
        result = Interactions_state(int_part)
        final_list.insert(0,result())
            
        if isinstance(result(),Particle):
            self.current_particle_list = final_list
            self.history_of_particle_lists.append(final_list)
    
        self.history_of_operations.append(result())
        
        if len(operation)==2:
            self.current_operation = next(self.op_ite)
        
    
    def get_matrix_element(self):
        """
        Calls recursively the method do_next_operation() until the matrix element is obtained.
        
        Returns
        ----------
        complex
            The matrix element from the Feynman Diagram.   
        """
        for n in range(self.number_of_operations):
            self.do_next_operation()
        return self.history_of_operations[-1]

class ComputeGraph:
    
    def __init__(self,particle_list,operations_list):
        self.particle_list = particle_list
        self.operations_list = operations_list
        
        self.FD_list = []
        self.DAG = nx.DiGraph()
        
        
    @classmethod
    def generate(cls,particle_list):
        operation_list = branch_calculator(particle_list)
        return cls(particle_list,operation_list)
    
    def FD_generator(self):
        self.FD_list = [None] * len(self.operations_list)
        for i in range(len(self.FD_list)):
            operations = list(self.operations_list[i].values())[0]
            FD = FeynmanDiagrams(self.particle_list,operations)
            graph=FD.get_FD()
            self.FD_list[i] = graph
        
    def FD_draw_all(self):
        self.FD_generator()
        for i in self.FD_list:
            plt.tight_layout()
            nx.draw_networkx(i, arrows=True)
            plt.show()
    
    def DAG_nodes(self,operations):
        FD = FeynmanDiagrams(self.particle_list,operations)
        FD.get_FD()
        nodes = []
        for i in FD.history_of_particle_lists:
            gen_nodes = []
            for j in i:
                gen_nodes.append(j.name)
            nodes.append(gen_nodes)
        nodes.append([FD.history_of_operations[-1]])
        return nodes

    def DAG_generator(self,i=None):
        if i is None:
            branches = self.operations_list.copy()
            color = ["cyan","cyan","lime"]
            size = [300, 300,600]
            line_style = ["black","black"]
            
        else:
            highlighted = self.operations_list[i]
            rest = self.operations_list.copy()
            rest.pop(i)
            rest.append(highlighted)
            branches = rest
            color = ["cyan","red","lime"]
            size = [300, 500,600]
            line_style = ["silver","black"]
        
        k=0
        mat_element=[]
                
        for operation in branches:
            if operation==branches[-1]:
                k=1
            
            branch=list(operation.values())[0]
            nodes = self.DAG_nodes(branch)
            
            for j in nodes:
                self.DAG.add_nodes_from(j,att_col=color[k],att_size=size[k])

            p=0
            for m in branch:
                for n in m:
                    self.DAG.add_edge(nodes[p][n],nodes[p+1][0],att_style = line_style[k])
                p+=1
            mat_element.append((nodes[-1][0],"+"))
           
        self.DAG.add_node("+",att_col=color[-1],att_size=size[-1])
        self.DAG.add_edges_from(mat_element[0:-1],att_style = line_style[0])
        self.DAG.add_edge(mat_element[-1][0],mat_element[-1][1],att_style = line_style[1])
        return self.DAG
        
    def draw_DAG(self,i=None):
        f = plt.figure(); f.set_figwidth(20); f.set_figheight(10) 
        
        if i is None:
            j=None
        else:
            j=i
            
        self.DAG_generator(j)
        
        color_map = nx.get_node_attributes(self.DAG,"att_col")
        colors = [color_map.get(node) for node in self.DAG.nodes()]

        size_list = nx.get_node_attributes(self.DAG,"att_size")
        sizes = [size_list.get(node) for node in self.DAG.nodes()]
        
        line_styles_list = nx.get_edge_attributes(self.DAG,"att_style")
        line_styles = [line_styles_list.get(edge) for edge in line_styles_list]
        
        nx.draw_networkx(self.DAG, arrows=True,node_color = colors, node_size = sizes, edge_color = line_styles)
        plt.tight_layout()
        plt.show()