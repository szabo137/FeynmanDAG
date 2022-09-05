"""
This is part of `FeynmanDAG` and it contains definition of the particles.
Currently, only the particles of the ABC model are supported.
"""

from __future__ import annotations

import numpy as np

from . import kinematics as kin

__all__ = ["Particle","ParticleA", "ParticleB", "ParticleC", "ParticleA_state", "ParticleB_state", "ParticleC_state"]

class Particle:  
    """ 
    A class used to represent the particles.
    
    ...
    
    Attributes
    ----------
    name : float
        The name of the particle.
    is_inc : bool
        Direction of the particle: incoming (True) or outgoing (False).
    is_real: bool
        Identity of the particle: real (True) or virtual (False).
    
    """
    
    def __init__(self, name, is_inc, is_real):
        
        """
        Constructs a particle with the minimum attributes necessary for its description.

        Parameters
        ----------
            name : float
                The name of the particle.
            is_inc : bool
                Direction of the particle: incoming (True) or outgoing (False).
            is_real: bool
                Identity of the particle: real (True) or virtual (False).

        """
        
        self.name = name
        self.is_real = is_real
        self.is_inc = is_inc


class ParticleA(Particle):
    """ 
    A class used to represent the particles of type A.
    
    ...
    
    Attributes
    ----------
    name : float
        The name of the particle.
    is_inc : bool
        Direction of the particle: incoming (True) or outgoing (False).
    is_real: bool
        Identity of the particle: real (True) or virtual (False).
    
    
    Methods:
    ----------
    mass()
        Returns the mass associated to particles of type A.
    
    """
    
    def __init__(self,*args):
        """
        Constructs a particle A with the minimum attributes necessary for its description.

        Parameters
        ----------
            name : float
                The name of the particle.
            is_inc : bool
                Direction of the particle: incoming (True) or outgoing (False).
            is_real: bool
                Identity of the particle: real (True) or virtual (False).
                
        """
        Particle.__init__(self,*args)
        
    @property
    def mass(self):
        """
        Returns the mass associated to particles of type A.
        
        Returns
        ----------
        float
            The mass associated to particles of type A.
            
        """
        return 2.0
    
class ParticleB(Particle):
    """ 
    A class used to represent the particles of type B.
    
    ...
    
    Attributes
    ----------
    name : float
        The name of the particle.
    is_inc : bool
        Direction of the particle: incoming (True) or outgoing (False).
    is_real: bool
        Identity of the particle: real (True) or virtual (False).
    
    
    Methods:
    ----------
    mass()
        Returns the mass associated to particles of type B.
    
    """
    
    def __init__(self,*args):
        """
        Constructs a particle B with the minimum attributes necessary for its description.

        Parameters
        ----------
            name : float
                The name of the particle.
            is_inc : bool
                Direction of the particle: incoming (True) or outgoing (False).
            is_real: bool
                Identity of the particle: real (True) or virtual (False).
            
        """
        Particle.__init__(self,*args)
        
    @property
    def mass(self):
        """
        Returns the mass associated to particles of type B.
        
        Returns
        ----------
        float
            The mass associated to particles of type B.
        """
        return 1.0

class ParticleC(Particle):
    """ 
    A class used to represent the particles of type C.
    
    ...
    
    Attributes
    ----------
    name : float
        The name of the particle.
    is_inc : bool
        Direction of the particle: incoming (True) or outgoing (False).
    is_real: bool
        Identity of the particle: real (True) or virtual (False).
    
    
    Methods:
    ----------
    mass()
        Returns the mass associated to particles of type A.
    
    """
    def __init__(self,*args):
        """
        Constructs a particle C with the minimum attributes necessary for its description.

        Parameters
        ----------
            name : float
                The name of the particle.
            is_inc : bool
                Direction of the particle: incoming (True) or outgoing (False).
            is_real: bool
                Identity of the particle: real (True) or virtual (False).
    
        """
        Particle.__init__(self,*args)
    
    @property
    def mass(self):
        """
        Returns the mass associated to particles of type C.
        
        Returns
        ----------
        float
            The mass associated to particles of type C.
            
        """
        return 0.5

class ParticleA_state(ParticleA):
    """ 
    A class used to represent the particles A and its states.
    
    ...
    
    Attributes
    ----------
    name : float
        The name of the particle.
    is_inc : bool
        Direction of the particle: incoming (True) or outgoing (False).
    is_real: bool
        Identity of the particle: real (True) or virtual (False).
    mom: float
        The modulus of the momentum of the particle.
    state: complex
        The state of the particle. By default it is the Feynman rule associated to external legs.
    
    
    Methods:
    ----------
    mass()
        Returns the mass associated to particles of type A.
    
    propagator()
        Returns the propagator associated to particles of type A.
        
    state()
        Returns the correct state associated to the particle that has been constructed.
        
    signed_mom()
        Returns the modulus of the momentum with its correct sign.
    
    """
    
    def __init__(self,name, is_inc, is_real, mom, state=None):
        """
        Constructs a particle A with the attributes necessary for its description and its state.

        Parameters
        ----------
            name : float
                The name of the particle.
            is_inc : bool
                Direction of the particle: incoming (True) or outgoing (False).
            is_real: bool
                Identity of the particle: real (True) or virtual (False).
            mom: float
                Modulus of the momentum of the particle.
            state: complex
                The state of the particle. By default it is the Feynman rule associated to external legs.
    
        """
        
        ParticleA.__init__(self, name, is_inc, is_real)
        self.mom = mom
        
        if state is None:
            if self.is_real:
                self.__state = 1
            else:
                raise ArgumentError("Virtual particle needs to have state")
        else:
            self.__state = state
    
    @property
    def propagator(self):
        """
        Returns the propagator associated to particles of type A.
        
        Returns
        ----------
        Complex
            The propagator associated to particles of type A.
        
        Raises
        ----------
        ArgumentError
            If the particle considered is real (is_real = True).
        
        """
        if not self.is_real:
            return 1j/((self.mom**2-self.mass**2))
        else:
            raise ArgumentError("External particle does not have propagator")
    
    @property
    def state(self):
        """
        Returns the proper state associated to particles of type A.
        
        Returns
        -------
        Complex
            The state associated of particles A depending on the particle type: real or virtual.
        
        """
        if self.is_real:
            return self.__state
        else:
            return self.__state * self.propagator
    
    @property
    def signed_mom(self):
        """
        Returns the modulus of the momentum with its correct sign.
        
        Returns
        ----------
        float
            The modulus of the momentum with its correct sign associated of particles: +/- is is_inc is True/False.
        
        """
        if self.is_inc:
            return self.mom
        else:
            return -self.mom
        
class ParticleB_state(ParticleB):
    """ 
    A class used to represent the particles B and its states.
    
    ...
    
    Attributes
    ----------
    name : float
        The name of the particle.
    is_inc : bool
        Direction of the particle: incoming (True) or outgoing (False).
    is_real: bool
        Identity of the particle: real (True) or virtual (False).
    mom: float
        The modulus of the momentum of the particle.
    state: complex
        The state of the particle. By default it is the Feynman rule associated to external legs.
    
    
    Methods:
    ----------
    mass()
        Returns the mass associated to particles of type B.
    
    propagator()
        Returns the propagator associated to particles of type B.
        
    state()
        Returns the correct state associated to the particle that has been constructed.
        
    signed_mom()
        Returns the modulus of the momentum with its correct sign.
    
    """
    
    def __init__(self,name, is_inc, is_real, mom, state=None):
        """
        Constructs a particle B with the attributes necessary for its description and its state.

        Parameters
        ----------
            name : float
                The name of the particle.
            is_inc : bool
                Direction of the particle: incoming (True) or outgoing (False).
            is_real: bool
                Identity of the particle: real (True) or virtual (False).
            mom: float
                Modulus of the momentum of the particle.
            state: complex
                The state of the particle. By default it is the Feynman rule associated to external legs.
    
        """
        ParticleB.__init__(self, name, is_inc, is_real)
        self.mom = mom
        
        if state is None:
            if self.is_real:
                self.__state = 1
            else:
                raise ArgumentError("Virtual particle needs to have state")
        else:
            self.__state = state
    
    @property
    def propagator(self):
        """
        Returns the propagator associated to particles of type B.
        
        Returns
        ----------
        Complex
            The propagator associated to particles of type B.
        
        Raises
        ----------
        ArgumentError
            If the particle considered is real (is_real = True).
        
        """
        if not self.is_real:
            return 1j/((self.mom**2-self.mass**2))
        else:
            raise ArgumentError("External particle does not have propagator")
    
    @property
    def state(self):
        """
        Returns the proper state associated to particles of type B.
        
        Returns
        ----------
        Complex
            The state associated of particles B depending on the particle type: real or virtual.
        
        """
        if self.is_real:
            return self.__state
        else:
            return self.__state * self.propagator
    
    @property
    def signed_mom(self):
        """
        Returns the modulus of the momentum with its correct sign.
        
        Returns
        ----------
        float
            The modulus of the momentum with its correct sign associated of particles: +/- is is_inc is True/False.
        
        """
        if self.is_inc:
            return self.mom
        else:
            return -self.mom
        
class ParticleC_state(ParticleC):
    """ 
    A class used to represent the particles C and its states.
    
    ...
    
    Attributes
    ----------
    name : float
        The name of the particle.
    is_inc : bool
        Direction of the particle: incoming (True) or outgoing (False).
    is_real: bool
        Identity of the particle: real (True) or virtual (False).
    mom: float
        The modulus of the momentum of the particle.
    state: complex
        The state of the particle. By default it is the Feynman rule associated to external legs.
    
    
    Methods:
    ----------
    mass()
        Returns the mass associated to particles of type C.
    
    propagator()
        Returns the propagator associated to particles of type C.
        
    state()
        Returns the correct state associated to the particle that has been constructed.
        
    signed_mom()
        Returns the modulus of the momentum with its correct sign.
    
    """
    
    def __init__(self,name, is_inc, is_real, mom, state=None):
        """
        Constructs a particle C with the attributes necessary for its description and its state.

        Parameters
        ----------
            name : float
                The name of the particle.
            is_inc : bool
                Direction of the particle: incoming (True) or outgoing (False).
            is_real: bool
                Identity of the particle: real (True) or virtual (False).
            mom: float
                Modulus of the momentum of the particle.
            state: complex
                The state of the particle. By default it is the Feynman rule associated to external legs.
    
        """
        ParticleC.__init__(self, name, is_inc, is_real)
        self.mom = mom
        
        if state is None:
            if self.is_real:
                self.__state = 1
            else:
                raise ArgumentError("Virtual particle needs to have state")
        else:
            self.__state = state
    
    @property
    def propagator(self):
        """
        Returns the propagator associated to particles of type C.
        
        Returns
        ----------
        Complex
            The propagator associated to particles of type C.
        
        Raises
        ----------
        ArgumentError
            If the particle considered is real (is_real = True).
        
        """
        if not self.is_real:
            return 1j/((self.mom**2-self.mass**2))
        else:
            raise ArgumentError("External particle does not have propagator")
    
    @property
    def state(self):
        """
        Returns the proper state associated to particles of type C.
        
        Returns
        ----------
        Complex
            The state associated of particles C depending on the particle type: real or virtual.
        
        """
        if self.is_real:
            return self.__state
        else:
            return self.__state * self.propagator
    
    @property
    def signed_mom(self):
        """
        Returns the modulus of the momentum with its correct sign.
        
        Returns
        ----------
        float
            The modulus of the momentum with its correct sign associated of particles: +/- is is_inc is True/False.
        
        """
        if self.is_inc:
            return self.mom
        else:
            return -self.mom