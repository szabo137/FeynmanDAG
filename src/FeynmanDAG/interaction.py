"""
This module is part of `FeynmanDAG` and it contains the definition of the interactions in the ABC model.
Currently, only the ABC interaction is supported.
"""

from __future__ import annotations

from .particles import ParticleA, ParticleB, ParticleC

__all__ = ["Interactions", "Interactions_FD", "Interactions_state"]

class Interactions:
    """ 
    A class used to represent the interactions between the particles.
    
    ...
    
    
    Attributes
    ----------
    particles : list
        A list that contains of 2 or 3 particles that will interact.
    
    Methods:
    ----------
    Number_of_particles()
        Returns the number of particles that interact.
    
    Number_of_output_particles()
        Returns the number of particles that arises after the interaction.
    
    """    
    
    def __init__(self,particles):
        """
        Constructs an object with the particles that will interact
        
        Parameters
        ----------
            particles : list
                A list that contains of 2 or 3 particles that will interact.            
            
        """
        self.particles = particles
    
    @property
    def Number_of_particles(self):
        """
        Returns the number of particles that interact.
        
        Returns
        ----------
        int
            Number of particles that interact: 2 or 3.
            
        """
        return len(self.particles)
    
    @property
    def Number_of_output_particles(self):
        """
        Returns the number of particles that arises after the interaction.
        
        Returns
        -------
        int
            Number of particles that arises from the interaction: 0 or 1.
        
        Raises
        ------
        ArgumentError
            If the number of particles that interact is different from 2 or 3.
        
        """
        if self.Number_of_particles == 2:
            return 1
        if self.Number_of_particles == 3:
            return 0
        raise ArgumentError(f"Only 2 and 3 particles are allowed to interact. Given {self.Number_of_particles}")


class Interactions_FD(Interactions):
    """ 
    A class used to represent the interactions between the particles and build the Feynman Diagrams.
    
    ...
    
    
    Attributes
    ----------
    particles : list
        A list that contains of 2 or 3 particles that will interact.
    
    
    Class attributes
    ----------
    OUTPUT_PARTICLE_FD: dict
        The keys correspond to the two particles that interact and the values to the product of the interaction.
        The particles are described within the ParticleA, ParticleB and ParticleC classes.
    
    
    Methods:
    ----------
    
    __Interact_ABC_2to1_FD(): 
        Returns the particle that arises from the interaction of 2 particles.
    
     __Interact_ABC_3to0_FD(): 
        Returns the expression for matrix element that arises from the interaction of 3 particles in the last step.
    
    """ 

    OUTPUT_PARTICLE_FD = { (ParticleA,ParticleB) : (ParticleC, "C"), (ParticleB,ParticleA) : (ParticleC, "C"),
                           (ParticleA,ParticleC) : (ParticleB, "B"), (ParticleC,ParticleA) : (ParticleB, "B"), 
                           (ParticleB,ParticleC) : (ParticleA, "A"), (ParticleC,ParticleB) : (ParticleA, "A") }
    
    
    def __init__(self,particles):
        """
        Constructs an object with the particles that will interact
        
        Parameters
        ----------
            particles : list
                A list that contains of 2 or 3 particles that will interact.            
            
        """
        Interactions.__init__(self, particles)
    
    def __call__(self):
        """
        Calls the correspondant method depending on the number of particles interacting.
        
        Parameters
        ----------
            particles : list
                A list that contains of 2 or 3 particles that will interact. 
        
        Returns
        ----------
            Calls __Interact_ABC_2to1_FD() method if 2 particles interact.
            
            Calls __Interact_ABC_3to0_FD() method if 3 particles interact.
        
        Raises
        ----------
        ArgumentError
            If the number of particles that interact is different from 2 or 3.
            
        """
        
        if self.Number_of_particles == 2:
            return self.__Interact_ABC_2to1_FD()
        if self.Number_of_particles == 3:
            return self.__Interact_ABC_3to0_FD()
        raise ArgumentError(f"Only 2 and 3 particles are allowed to interact. Given {self.Number_of_particles}")
    
    
    def __Interact_ABC_2to1_FD(self):
        """
        It gives the particle that arises when 2 particles interact in a vertex through the ABC interaction.
        
        Steps:
            1st. Check if the particles considered are different.
            2nd. Calculates the particle that will be produced.
            3rd. Associates the correct attributes and gives a name that is inheritated from the parent particles.
            
        Returns
        ----------
        Object from ParticleA, ParticleB or ParticleC classes.    
            The particle with the correct attributes that has been produced.

        Raises
        ----------
        ArgumentError
            If the particles are from the same type, the interaction is not allowed.
            
        """
        
        p1 = self.particles[0]
        p2 = self.particles[1]
        
        if (p1.__class__, p2.__class__) in self.OUTPUT_PARTICLE_FD.keys():
            p3_type = self.OUTPUT_PARTICLE_FD[(p1.__class__, p2.__class__)][0]
            p3_type_str = self.OUTPUT_PARTICLE_FD[(p1.__class__, p2.__class__)][1]
            p3 = p3_type(p3_type_str+"("+p1.name+","+p2.name+")",True,False)
            return p3
        else:
            raise ArgumentError(f"Interaction of {p1.name} and {p2.name} is not allowed.")
    
    
    def __Interact_ABC_3to0_FD(self):
        """
        It gives the matrix element that arises when 3 particles interact in a vertex through the ABC interaction.
        
        Steps
        ----------
            1st. Check if all the particles considered are different.
            2nd. Returns a string with the expression of the matrix element.
            
        Returns
        ----------
        str
            The expression for the matrix element of a given process.

        Raises
        ----------
        ArgumentError
            If the particles are from the same type, the interaction is not allowed.
            
        """
        
        p1 = self.particles[0]
        p2 = self.particles[1]
        p3 = self.particles[2]
        
        check_list = [ParticleA,ParticleB,ParticleC]
        inp_list = [p1,p2,p3]

        for p in inp_list:
            if p.__class__ in check_list:
                check_list.remove(p.__class__)

        if len(check_list) == 0:
            return "M("+p1.name+","+p2.name+","+p3.name+")"
        else:
            raise ArgumentError("Only 3 particles of the different type can interact.")

class Interactions_state(Interactions):
    
    """ 
    A class used to represent the interactions between the particles and build the matrix elements.
    
    ...
    
    
    Attributes
    ----------
    particles : list
        A list that contains of 2 or 3 particles that will interact.
    
    
    Class attributes
    ----------
    LAMBDA_ABC: float
        Coupling constant of the interaction associated to ABC vertex.
    
    OUTPUT_PARTICLE_STATE: dict
        The keys correspond to the two particles that interact and the values to the product of the interaction.
        The particles are described within the ParticleA_state, ParticleB_state and ParticleC_state classes.
    
    
    Methods:
    ----------
    output_type():
        Returns the type of output after from the interaction.
    
    __Interact_ABC_2to1_state(): 
        Returns the particle that arises from the interaction of 2 particles.
    
     __Interact_ABC_3to0_state(): 
        Returns the matrix element that arises from the interaction of 3 particles in the last step.
    
    """
    
    LAMBDA_ABC = 0.1
    
    OUTPUT_PARTICLE_STATE = { (ParticleA_state, ParticleB_state) : (ParticleC_state, "C"), 
                              (ParticleB_state, ParticleA_state) : (ParticleC_state, "C"), 
                              (ParticleA_state, ParticleC_state) : (ParticleB_state, "B"), 
                              (ParticleC_state, ParticleA_state) : (ParticleB_state, "B"), 
                              (ParticleB_state, ParticleC_state) : (ParticleA_state, "A"), 
                              (ParticleC_state, ParticleB_state) : (ParticleA_state, "A"), }
    
    def __init__(self,particles):
        """
        Constructs an object with the particles that will interact
        
        Parameters
        ----------
            particles : list
                A list that contains of 2 or 3 particles that will interact.            
            
        """
        Interactions.__init__(self, particles)
    
    def __call__(self):
        """
        Calls the correspondant method depending on the number of particles interacting.
        
        Parameters
        ----------
            particles : list
                A list that contains of 2 or 3 particles that will interact. 
        
        Returns
        ----------
            Calls __Interact_ABC_2to1_state() method if 2 particles interact.
            
            Calls __Interact_ABC_3to0_state() method if 3 particles interact.
        
        Raises
        ----------
        ArgumentError
            If the number of particles that interact is different from 2 or 3.
            
        """
        
        if self.Number_of_particles == 2:
            return self.__Interact_ABC_2to1_state()
        if self.Number_of_particles == 3:
            return self.__Interact_ABC_3to0_state()
        raise ArgumentError(f"Only 2 and 3 particles are allowed to interact. Given {self.Number_of_particles}")
    
    @property
    def output_type(self):
        """
        It gives the output type that arises from the interaction.
        
        Steps:
            1st. Check if all the particles considered are different.
            2nd. Returns a string with the expression of the matrix element.
            
        Returns
        ----------
        Object from ParticleA_state, ParticleB_state or ParticleC_state classes
            If the number of output particles is 1.
        complex
            If the number of output particles is 0.

        Raises
        ----------
        ArgumentError
            If the particles that arises are different from 0 or 1.
            
        """
        if self.Number_of_output_particles == 1:
            return self.OUTPUT_PARTICLE_STATE[(self.particles[0].__class__, self.particles[1].__class__)]
        if self.Number_of_output_particles == 0:
            return complex
        raise ArgumentError(f"Only a particle (1) or the matrix element (0) are allowed to arises from the interaction. Given {self.Number_of_output_particles}")
    
    def __Interact_ABC_2to1_state(self):
        """
        It gives the particle that arises when 2 particles interact in a vertex through the ABC interaction.
        
        Steps:
            1st. Check if the particles considered are different.
            2nd. Calculates the particle that will be produced.
            3rd. Associates the correct attributes and gives a name that is inheritated from the parent particles.
        
        The state of the new particle is the product of: the coupling constant from ABC interaction and the product
        of the states from the 2 particles that are interacting. 
        
        Returns
        ----------
        Object from ParticleA, ParticleB_state or ParticleC_state classes.    
            The particle with the correct attributes and state that has been produced.

        Raises
        ----------
        ArgumentError
            If the particles are from the same type, the interaction is not allowed.
            
        """
        
        p1 = self.particles[0]
        p2 = self.particles[1]
        if (p1.__class__, p2.__class__) in self.OUTPUT_PARTICLE_STATE.keys():
            p3_type = self.OUTPUT_PARTICLE_STATE[(p1.__class__, p2.__class__)][0]
            p3_type_str = self.OUTPUT_PARTICLE_STATE[(p1.__class__, p2.__class__)][1]
            p3_mom = p1.signed_mom + p2.signed_mom
            p3_state = -1j*self.LAMBDA_ABC*p1.state*p2.state
            p3 = p3_type(p3_type_str+"("+p1.name+","+p2.name+")",True,False,p3_mom,p3_state)
            return p3
        else:
            raise ArgumentError(f"Interaction of {p1.__class__} and {p2.__class__} is not allowed.")
    
    def __Interact_ABC_3to0_state(self):
        """
        It gives the matrix element that arises when 3 particles interact in a vertex through the ABC interaction.
        
        Steps
        ----------
            1st. Check if all the particles considered are different.
            2nd. Returns a complex number with the expression of the matrix element.
        
        The expression for the matrix element is the product of the coupling constant and the three states of the
        particles that are interacting.
            
        Returns
        ----------
        complex
            The the matrix element of a given process.

        Raises
        ----------
        ArgumentError
            If the particles are from the same type, the interaction is not allowed.
            
        """
        
        p1 = self.particles[0]
        p2 = self.particles[1]
        p3 = self.particles[2]
        
        LAMBDA_ABC = 0.1
        check_list = [ParticleA_state,ParticleB_state,ParticleC_state]
        inp_list = [p1,p2,p3]

        for p in inp_list:
            if p.__class__ in check_list:
                check_list.remove(p.__class__)

        if len(check_list) == 0:
            return p1.state*p2.state*p3.state*-1j*self.LAMBDA_ABC
        else:
            raise ArgumentError("Only 3 particles of the different type can interact.")
