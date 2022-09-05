"""
This module is part of FeynmanDAG and it provides functionality to produce Feynman diagrams and computation graphs from external particles.

Currently, only the ABC model with one vertex is supported.
"""

__all__ = ["filter_ABC", "max_comb", "comb", "comb_rec", "branch_calculator"]

def filter_ABC(particle,particles):
    """ 
        It filts out the particles of a list that cannot be combined with a given particle.
        
        ABC filter:
            The particles that are subtracted are those whose are the same type of the considered particle.
        
        Parameters
        ----------
        particle: object from ParticleA, ParticleB or ParticleC classes 
            The particle that will filter the rest of particles from the list.
            
        particles : list
            The list that contains the particles that will be filtered.
        
        Returns
        ----------
        int
            The position of the particle in the input list.
    """
    
    filt_list = list(filter(lambda i: i.__class__ != particle.__class__, particles))
    filt_list.insert(0,particle)
    return filt_list


def max_comb(particles):
    """ 
        It gives the position of the particle from a given list that leads to the highest number of combinations
        with the rest of particles.
        
        Parameters
        ----------
        particles : list
            The list that contains the particles that will be combined.
            
        Steps
        ---------
            1st. It calculates the combinations of each particle with the rest.
            2nd. Saves the number of combinations on a list.
            3rd. Finds the position of the list with the highest number.
            4th. Returns the correspondent particle.
        
        Returns
        ----------
        int
            The position of the particle in the input list.
    """

    number_of_combinations=[]
    for k in particles:
        filtered_list = filter_ABC(k,particles)
        combinations_from_filtered_list = list(combinations(filtered_list,2))
        number_of_combinations.append(len(combinations_from_filtered_list))

    position = np.where(np.array(number_of_combinations) == np.max(number_of_combinations))[0][0]
    return position


def comb(old_particles,operations):
    """
    From a list of particles, the function returns a set of new list of particles where 2 of them from the
    original list have been combined. 
    
    Parameters
    ----------
    old_particles: list/dict
        The list that contains the particles that will be combined.
        The dictionary whose key and value contains the name of the branch and the list of particles, 
        respectively.
    
    operations: list/dict
        The list the position of the particles that have combined. By default is an empty list: [].
        The dictionary whose key and value contains the name of the branch and the list of previous 
        operations, respectively.
    
    Steps
    ----------
        1st. It identifies if the parameters as passed out as list or dict. 
        2nd. If the number of particles is 2:
                1st. Call the function max_comb(particles) to get the particle with the highest number of 
                     combinations.
                2nd. Does all the combinations and saves the new branches that have been produced with the
                     new list of particles and the operation that has been realized.
             If the number of particles is 3:
                Returns the expression for the matrix element and add (0,1,2) to the operation list.
    
    Returns
    ----------
    new_particles: list
        The list of branches with all the list of particles that have been generated.
        
    positions: list
        The list of branches with all the operations that lead to a combination and generate a new list 
        of particles.
    
    """

    if type(old_particles)==list:
        particles = old_particles
        gen="branch "
        old_values=[]

    if type(old_particles)==dict:
        particles = list(old_particles.values())[0]
        gen = list(old_particles.keys())[0]
        old_values=list(operations.values())[0]

    positions=[]
    new_particles=[]

    index = max_comb(old_particles)

    if len(particles)!=3:

        particle = particles[index]
        inter = filter_ABC(particle,particles)
        
        part_comb=list(combinations(inter,2))
        part_comb_filt=part_comb[0:(len(inter)-1)]

        for i in range(len(part_comb_filt)):

            history_of_operations = old_values.copy()
            particles_copy = particles.copy()

            keys=gen+str(i+1)+"."

            values = (particles_copy.index(part_comb_filt[i][0]),particles_copy.index(part_comb_filt[i][1]))
            history_of_operations.append(values)
            positions.append({keys:history_of_operations})

            new_particle = Interactions_FD([part_comb_filt[i][0],part_comb_filt[i][1]])

            particles_copy.remove(part_comb_filt[i][0])
            particles_copy.remove(part_comb_filt[i][1])
            particles_copy.insert(0,new_particle())

            new_particles.append({keys:particles_copy})

    if len(particles)==3:
        history_of_operations = old_values.copy()
        particles_copy = particles.copy()

        keys=gen
        values=(0,1,2)
        history_of_operations.append(values)
        positions.append({keys:history_of_operations})

        new_particle = Interactions_FD(particles_copy)
        new_particles.append({keys:new_particle()})

    return new_particles, positions

def comb_rec(list_of_particles,list_of_operations):
    """
    From a list of branches, it calculates a set of new branches for each branch of the initial list.
    
    Parameters
    ----------
    list_of_particles: list
        The list that contains the branches with the different list of particles that have been combined previously.
        Its elements are dictionaries whose key and value contains the name of the branch and the list of particles, 
        respectively.
    
    list_of_operations: list
        The list that contains the branches with the different position of the particles that have combined previously.
        Its elements dictionaries key and value contains the name of the branch and the list of previous operations, 
        respectively.
    
    Steps
    ----------
        1st. It sees if any operation has been performed previously. 
             If not, it does the first iteration.
             If yes:
        2nd. It calculates the next combinations by calling the function comb() in each branch.
    
    Returns
    ----------
    new_particles: list
        The list of branches with all the list of particles that have been generated.
        
    new_operations: list
        The list of branches with all the new operations that lead to a combination and generate a new list 
        of particles.
    
    """
    if list_of_operations==[]:
        new_particles = comb(list_of_particles,list_of_operations)[0]
        new_operations = comb(list_of_particles,list_of_operations)[1]

    else:

        new_particles = []
        new_operations = []

        for i in range(len(list_of_particles)):
            new_part, new_op = comb(list_of_particles[i],list_of_operations[i])
            for j in range(len(new_op)):
                new_particles.append(new_part[j])
                new_operations.append(new_op[j])

        if [] in new_particles and [] in new_operations:
            new_particles.remove([])
            new_operations.remove([])

    return new_particles,new_operations

def branch_calculator(particles):
    """
    From a list of particles, it calculates all the branches that lead to a Feynman diagram.
    
    Parameters
    ----------
    particles: list
        The list that contains the particles of the process.
    
    Steps
    ----------
        1st. It creates a list called FD where the combinations that have to be made between the particles in
             order to obtain the Feynman Diagrams will be stored.
        2nd. Since the particles are combined in pairs until there are 3 particles left, the number of operations
             that must be done is # op = # of particles - 2, therefore, it calls the function comb_rec() recursively
             # op times.
    
    Returns
    ----------
    op: list
        The list of branches with all the operations that generate a Feynman Diagram.
        
    """
    
    FD=[]
    for i in range(len(particles)-2):
        new_part = comb_rec(particles,FD)[0]
        new_operations = comb_rec(particles,FD)[1]
        particles = new_part
        FD = new_operations
    return FD