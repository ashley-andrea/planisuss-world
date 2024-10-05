import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import time
import math
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk

# World constants
NUMCELLS = 50  # Size of the grid
NUMDAYS = random.randint(50, 500)  # Number of days the simulation will run

# Species constants: the following parameters are customizable in the GUI
MAX_HERD = 100  # Maximum number of Erbast in a cell
MAX_PRIDE = 100  # Maximum number of Carviz in a cell
AGING_ERBAST = 5  # Energy lost each month (10 days)
AGING_CARVIZ = 5  # Energy lost each month
VEGETOB_GROWTH = 2  # Vegetob growth rate
NEIGHBORHOOD = 4  # Neighborhood size, that is, how many cells around are evaluated
LIFETIME_ERBAST = 25  # Erbast lifetime
LIFETIME_CARVIZ = 30  # Carviz lifetime
ENERGY_THRESHOLD_ERBAST = 5  # Erbast energy threshold for moving   
ENERGY_THRESHOLD_CARVIZ = 2  # Carviz energy threshold for moving
SA_THRESHOLD_ERBAST = 0.2  # Erbast social attitude threshold for moving
SA_THRESHOLD_CARVIZ = 0.1  # Carviz social attitude threshold for moving

# World class
class World:
    def __init__(self, size, ground_ratio):  
        '''
        World initialization:
        - size: size of the world grid
        - ground_ratio: ratio of ground cells to total cells
        The __init__ function initializes the world grid, ground cells, and species.
        '''
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  
        self.ground_cells = int(ground_ratio * (size * size))  
        self.initialize_grid()

        # Initializing species
        self.vegetob = Vegetob(self)
        self.erbast = Erbast(self)
        self.carviz = Carviz(self)

        # Initializing day counter
        self.day = 0  

        # Placing species randomly in the world
        self.vegetob.place_randomly(int(0.8*self.ground_cells))  
        self.erbast.place_randomly(int(random.uniform(0.2, 0.5)*self.ground_cells))   
        self.carviz.place_randomly(int(random.uniform(0.2, 0.5)*self.ground_cells)) 

    def initialize_grid(self):
        ''' 
        Function to initialize the world grid. It creates a central block of ground cells and then expands it
        in layers of decreasing density. It also ensures cell connectivity by calling ensure_neighbors().
        '''
        # Setting the boundary cells to water
        self.grid[0, :] = self.grid[-1, :] = 0  
        self.grid[:, 0] = self.grid[:, -1] = 0  

        # Creating a central block of ground cells
        center = self.size // 2
        core_size = int(self.size * 0.45)  
        core_start = center - core_size // 2
        core_end = center + core_size // 2
        
        for i in range(core_start, core_end):
            for j in range(core_start, core_end):
                if self.ground_cells > 0:
                    self.grid[i, j] = 1
                    self.ground_cells -= 1

        # Expanding ground cells from center outward, more sparsely as we go
        for layer in range(1, (self.size - core_size) // 2):
            num_cells_in_layer = max(1, int((self.size - layer) * random.uniform(0.3, 0.6)))
            layer_cells = []

            # Defining the boundary of this layer
            for i in range(core_start - layer, core_end + layer):
                for j in range(core_start - layer, core_end + layer):
                    if 0 < i < self.size-1 and 0 < j < self.size-1 and self.grid[i, j] == 0:
                        layer_cells.append((i, j))

            # Randomly choosing some cells in this layer for ground
            selected_cells = random.sample(layer_cells, min(num_cells_in_layer, len(layer_cells)))
            for (i, j) in selected_cells:
                if self.ground_cells > 0:
                    self.grid[i, j] = 1
                    self.ground_cells -= 1

            self.ensure_neighbors()

    def count_neighbors(self, x, y):
        '''
        Counts the number of neighbors for a cell in the grid. 
        '''
        neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if 0 <= x + dx < self.size and 0 <= y + dy < self.size:
                    neighbors += self.grid[x + dx, y + dy]
        return neighbors

    def ensure_neighbors(self):
        '''
        Ensures that each ground cell has at least 3 neighbors.
        '''
        to_relocate = []

        # Checking for ground cells with less than 3 neighbors and marking them for relocation
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                if self.grid[i, j] == 1 and self.count_neighbors(i, j) < 3:
                    to_relocate.append((i, j))
                    self.grid[i, j] = 0  

        # Reallocating ground cells starting from the inner regions
        for (x, y) in to_relocate:
            while True:
                new_x = random.randint(1, self.size - 2)
                new_y = random.randint(1, self.size - 2)
                if self.grid[new_x, new_y] == 0 and self.count_neighbors(new_x, new_y) >= 3:
                    self.grid[new_x, new_y] = 1
                    break
    
    def get_neighborhood_positions(self, x, y, neighborhood_size):
        '''
        Returns the positions of the cells in the neighborhood of the cell (x, y).
        '''
        neighborhood = []
        for i in range(x - neighborhood_size, x + neighborhood_size + 1):
            for j in range(y - neighborhood_size, y + neighborhood_size + 1):
                if 1 <= i < self.size-1 and 1 <= j < self.size-1:  
                    neighborhood.append((i, j))
        return neighborhood

    def evaluate_cell(self, x, y, species_type):
        '''
        Evaluates the cell based on the species type.
        '''
        if species_type == 'Erbast':
            return self.vegetob.intensity.get((x, y), 0)  # Erbast prefers Vegetob-rich cells
        elif species_type == 'Carviz':
            return self.erbast_density((x, y))  # Carviz prefers Erbast-rich cells

    def erbast_density(self, pos):
        '''
        Returns the number of Erbast creatures in a given cell.
        '''
        x, y = pos
        return sum(1 for erbast in self.erbast.creature_pop if erbast['position'] == (x, y))
    
    def carviz_density(self, pos):
        '''
        Returns the number of Carviz creatures in a given cell.
        '''
        x, y = pos 
        return sum(1 for carviz in self.carviz.creature_pop if carviz['position'] == (x, y))

    def check_cell_limits(self, x, y):
        '''
        Ensures herd and pride limits for the cell.
        '''
        if self.erbast_density((x, y)) > MAX_HERD or self.carviz_density((x, y)) > MAX_PRIDE:
            return False  # Cell is full
        return True

    def is_valid_move(self, x, y):
        '''
        Ensures the move is within the grid limits and on a ground cell.
        '''
        if 0 <= x < self.size and 0 <= y < self.size:
            is_valid = self.grid[x, y] == 1
            return is_valid
        return False    

    def validate_creature_positions(self):
        '''
        Validates the creature positions. 
        Prints a warning if a creature is found at an invalid position.
        '''
        for species in [self.erbast, self.carviz]:
            species_name = species.__class__.__name__
            for creature in species.creature_pop:
                x, y = creature['position']
                if not self.is_valid_move(x, y):
                    print(f"Warning: {species_name} at invalid position {(x, y)}")
                    print(f"Grid value: {self.grid[x, y]}")
    
    def update(self):
        '''
        Updates the world for the next time step (day).
        '''
        self.day += 1
        print(f"Day {self.day}")

        # Moving creatures, grazing, and updating social groups.
        self.vegetob.grow()
        self.erbast.move()  
        self.erbast.graze()  
        self.carviz.move()  
        self.erbast.fuse_herds()
        self.carviz.fuse_prides_or_fight()
        self.carviz.pride_hunt(self.erbast)  
        self.erbast.spawn()
        self.carviz.spawn()
        self.erbast.overwhelmed()
        self.carviz.overwhelmed()
        self.validate_creature_positions()

        print(f"After update: {len(self.erbast.creature_pop)} Erbast and {len(self.carviz.creature_pop)} Carviz")

# Superclass
class Species:
    def __init__(self, world):
        '''
        Species initialization.
        - world: reference to the world object
        - positions: set of tuples representing positions on the grid
        - memory: dictionary to store memory data for strategic decisions
        - creature_pop: list of individual creature attributes per position
        '''
        self.world = world
        self.positions = set() 
        self.memory = {} 
        self.creature_pop = [] 

    def place_randomly(self, count, create_creature_data):
        '''
        Places creatures randomly in the world.
        '''
        placed = 0
        while placed < count:
            x, y = random.randint(1, self.world.size-2), random.randint(1, self.world.size-2)
            if self.world.is_valid_move(x, y) and self.world.check_cell_limits(x, y): 
                    creature_data = create_creature_data(self, x, y)
                    self.add_creature(creature_data)
                    self.positions.add((x, y))
                    placed += 1
        print(f"Placed {count} creatures of {self.__class__.__name__}")
    
    def add_creature(self, creature_data):
        '''
        Adds a creature to the creature population.
        '''
        self.creature_pop.append(creature_data)

    def group_by_position(self, filter_func=None, group_by_time=False):
        '''
        Group creatures by their positions, optionally filtering and grouping by time.
        
        - param filter_func: optional function to filter creatures before grouping
        - param group_by_time: if True, group by both position and move_time
        - return: dictionary with positions as keys and lists of creatures (or time-grouped creatures) as values
        '''
        groups = {}
        for creature in self.creature_pop:
            if filter_func is None or filter_func(creature):
                pos = tuple(creature['position'])
                if group_by_time:
                    move_time = creature.get('move_time', 0)
                    if pos not in groups:
                        groups[pos] = {}
                    if move_time not in groups[pos]:
                        groups[pos][move_time] = []
                    groups[pos][move_time].append(creature)
                else:
                    if pos not in groups:
                        groups[pos] = []
                    groups[pos].append(creature)
        return groups

    def move(self, evaluate_cell_with_memory, energy_threshold, social_attitude_threshold):
        '''
        Moves the creatures in the world. Decisions are initially made collectively by groups of creatures,
        but individual creatures may choose to move independently if they have enough energy and social attitude.
        Cells are evaluated based on their suitability, determined by the evaluate_cell_with_memory function.

        The neighborhood size determines the size of the area evaluated for decision making. Creatures are able
        to look past their immediate neighborhood to evaluate the larger area and follow a trajectory towards 
        the most suitable cell.
        '''
        to_remove = []
        groups_by_pos = self.group_by_position()

        new_pop = []

        for pos, group in groups_by_pos.items():
            x, y = pos

            info_neighborhood = self.world.get_neighborhood_positions(x, y, NEIGHBORHOOD)
            move_neighborhood = self.world.get_neighborhood_positions(x, y, 1)
            valid_moves = [
                pos for pos in move_neighborhood if self.world.is_valid_move(*pos) and self.world.check_cell_limits(*pos) 
                and pos not in group[0]['memory']['last_visited']]

            if valid_moves:
                cell_values = {pos: evaluate_cell_with_memory(group[0], pos) for pos in info_neighborhood}
                best_direction = max(cell_values, key=cell_values.get)
                best_position = max(valid_moves, key=lambda pos: (
                    abs(pos[0] - best_direction[0]) + abs(pos[1] - best_direction[1]),
                    cell_values.get(pos, float('-inf'))
                ))
                current_value = cell_values.get((x, y), float('-inf'))
                best_value = cell_values.get(best_direction, float('-inf'))
                group_decision = best_position if best_value > current_value else None

                for creature in group:
                    if creature['energy'] <= 0:
                        to_remove.append(creature)
                        continue

                    if group_decision:
                        if creature['energy'] >= energy_threshold or creature['social_attitude'] >= social_attitude_threshold:
                            creature['position'] = group_decision
                            creature['energy'] -= 1
                            creature['energy'] = max(creature['energy'], 0)
                            creature['memory']['last_visited'].add((x, y))
                            if hasattr(self, 'moved_erbast'):
                                self.moved_erbast.add(id(creature))  
                            if hasattr(creature, 'move_time'):
                                creature['move_time'] = self.world.day
                    else:
                        # Increasing the threshold ensures that individual creatures are less likely to move
                        if creature['energy'] >= energy_threshold * 3 and creature['social_attitude'] >= social_attitude_threshold * 3:
                            individual_best = max(valid_moves, key=lambda pos: cell_values.get(pos, float('-inf')))
                            creature['position'] = individual_best
                            creature['energy'] -= 1
                            creature['energy'] = max(creature['energy'], 0)
                            creature['memory']['last_visited'].add((x, y))
                            if hasattr(self, 'moved_erbast'):
                                self.moved_erbast.add(id(creature))  
                            if hasattr(creature, 'move_time'):
                                creature['move_time'] = self.world.day
                    new_pop.append(creature)
            else:
                for creature in group:
                    new_pop.append(creature)

        for member in to_remove:
            self.creature_pop.remove(member)
            if member['position'] in self.positions:
                self.positions.remove(member['position'])

        self.creature_pop = new_pop
        self.creature_pop = [creature for creature in self.creature_pop if self.world.is_valid_move(*creature['position'])]
        self.positions = set(tuple(creature['position']) for creature in self.creature_pop)

        if hasattr(self, 'moved_erbast'):
            return self.moved_erbast

    def spawn(self, aging, lifetime, max_size):
        '''
        Spawns new creatures upon death of the parent, distributing its attributes between the offspring.
        '''
        new_pop = []
        to_remove = []

        groups_by_pos = self.group_by_position()

        for pos, group in groups_by_pos.items():
            cell_new_pop = []
            for parent in group:
                parent['age'] += 1
                if parent['age'] % 10 == 0:
                    parent['energy'] -= aging
                    parent['energy'] = max(parent['energy'], 0)
                if parent['age'] == lifetime:
                    to_remove.append(parent)

                    random_distribution = random.random()

                    offspring = [parent.copy() for _ in range(2)]
                    for i, off in enumerate(offspring):
                        off.update({
                        'age': 0,
                        'energy': parent['energy'] // 2,
                        'social_attitude': parent['social_attitude'] * 2 * (random_distribution if i == 0 else 1-random_distribution),
                        'memory': parent['memory'].copy(),
                        'position': parent['position']
                    })

                    cell_new_pop.extend(offspring)
                else:
                    cell_new_pop.append(parent)

            cell_new_pop = cell_new_pop[:max_size]
            new_pop.extend(cell_new_pop)

        for parent in to_remove:
            self.creature_pop.remove(parent)
            if parent['position'] in self.positions:
                self.positions.remove(parent['position'])

        self.creature_pop = new_pop
        self.positions = set(member['position'] for member in self.creature_pop)

    def overwhelmed(self):
        '''
        Removes creatures from the world when their cell is completely eaten by Vegetob.
        '''
        to_remove = []

        for creature in self.creature_pop:
            x, y = creature['position']
            neighborhood_positions = self.world.get_neighborhood_positions(x, y, 1)

            if all(self.world.evaluate_cell(pos[0], pos[1], 'Erbast') == 100 for pos in neighborhood_positions):
                to_remove.append(creature)

        for member in to_remove:
            self.creature_pop.remove(member)
            if member['position'] in self.positions:
                self.positions.remove(member['position'])


# Subclasses
class Vegetob(Species):
    def __init__(self, world):
        '''
        Vegetob initialization.
        - intensity: dictionary to store the intensity of Vegetob in each cell
        '''
        super().__init__(world)
        self.intensity = {}  

    def place_randomly(self, count):
        '''
        Places Vegetob randomly in the world, initializing their attributes.
        '''
        def create_vegetob_data(self, x, y):
            return {
                'position': (x, y),
                'intensity': random.randint(0, 100)
            }
        super().place_randomly(count, create_vegetob_data)

    def add_creature(self, creature_data):
        '''
        Adds Vegetob to the creature population.
        '''
        pos = creature_data['position']
        self.intensity[pos] = creature_data['intensity']
        self.positions.add(pos)

    def grow(self):
        '''
        Simulates vegetob growth, increasing intensity for each cell it inhabits.
        '''
        for x in range(self.world.size):
            for y in range(self.world.size):
                pos = (x, y)
                if self.world.grid[pos] == 1:
                    if not pos in self.intensity:
                        self.intensity[pos] = 0

                    self.intensity[pos] += VEGETOB_GROWTH
                    self.intensity[pos] = min(100, self.intensity[pos])

                    if pos not in self.positions and self.intensity[pos] > 0:
                        self.positions.add(pos)

    def move(self):
        pass

    def spawn(self):
        pass

    def overwhelmed(self):
        pass

class Erbast(Species):
    def __init__(self, world):
        '''
        Erbast initialization.
        - moved_erbast: set to keep track of Erbast that moved
        - memory: strategic decisions are based on the last visited cells, and dangerous cells.
        '''
        super().__init__(world)        
        self.moved_erbast = set()  
        self.memory = {
            "last_visited": set(),  
            "dangerous_cells": set()  
        }

    def place_randomly(self, count):
        '''
        Places Erbast randomly in the world, initializing their attributes.
        '''
        def create_erbast_data(self, x, y):
            return {
                'position': (x, y),
                'energy': random.randint(50, 100),  
                'lifetime': LIFETIME_ERBAST,  
                'age': 0,  
                'social_attitude': random.uniform(0, 1),  
                'memory': {
                    "last_visited": set(),
                    "dangerous_cells": set()
                }
            }
        
        super().place_randomly(count, create_erbast_data)

    def add_creature(self, creature_data):
        '''
        Adds Erbast to the creature population.
        '''
        self.creature_pop.append(creature_data)

    def move(self):
        '''
        Moves the Erbast in the world.
        '''
        super().move(self.evaluate_cell_with_memory, ENERGY_THRESHOLD_ERBAST, SA_THRESHOLD_ERBAST)

    def evaluate_cell_with_memory(self, herd_member, position):
        '''
        Evaluates the cell based on Vegetob availability, depletion avoidance, and dangerous cells.
        '''
        vegetob_availability = 3 if self.world.evaluate_cell(position[0], position[1], 'Erbast') > 30 else 0
        depletion_penalty = 1 if self.world.evaluate_cell(position[0], position[1], 'Erbast') < 10 else 0
        dangerous_memory = 2 if position in herd_member['memory']['dangerous_cells'] else 0
        return max(0, vegetob_availability - dangerous_memory - depletion_penalty)
            
    def graze(self):
        '''
        Lets stationary Erbast graze on Vegetob, increasing energy and adjusting 
        social attitude based on availability.
        '''
        def stationary_filter(erbast):
            return id(erbast) not in self.moved_erbast
        
        herds_by_pos = self.group_by_position(filter_func=stationary_filter)

        for pos, herd in herds_by_pos.items():
            vegetob_density = self.world.evaluate_cell(pos[0], pos[1], 'Erbast')
            
            herd.sort(key=lambda member: member['energy'])

            for erbast in herd:
                if vegetob_density > 0:
                    energy_gain = min(1, vegetob_density)
                    erbast['energy'] += energy_gain
                    erbast['energy'] = min(erbast['energy'], 100)
                    vegetob_density -= energy_gain
                else:
                    break  
            
            self.world.vegetob.intensity[pos] = vegetob_density
            
            # Decreasing social attitude for Erbast that did not receive energy
            for erbast in herd:
                if vegetob_density <= 0 and erbast['energy'] < 50:  
                    erbast['social_attitude'] -= 0.1  
                    erbast['social_attitude'] = max(0, erbast['social_attitude'])  

    def fuse_herds(self):
        '''
        Fuses herds when multiple herds move to the same cell.
        '''
        herds_by_pos = self.group_by_position()
        new_pop = []

        for pos, herd_members in herds_by_pos.items():
            if len(herd_members) > 1:
                # Sorting herd members by energy to prioritize stronger members
                herd_members.sort(key=lambda x: x['energy'], reverse=True)

                # Fusing herds up to MAX_HERD size
                fused_herd = herd_members[:MAX_HERD]

                # Combining memories and adapting social attitudes
                combined_memory = {}
                for herd_member in fused_herd:
                    combined_memory.update(herd_member['memory'])
                    herd_member['social_attitude'] = sum([m['social_attitude'] for m in fused_herd]) / len(fused_herd)
                
                for herd_member in herd_members:
                    herd_member['memory'] = combined_memory

                new_pop.extend(fused_herd)

                # Handling excess members
                if not self.world.check_cell_limits(pos[0], pos[1]):
                    excess_members = herd_members[MAX_HERD:]
                    self.handle_excess_members(excess_members, pos)
            else:
                new_pop.extend(herd_members)
                
        self.creature_pop = new_pop
        self.positions = set(member['position'] for member in self.creature_pop)

    def handle_excess_members(self, excess_members, current_pos):
        '''
        Finds a nearby empty cell for excess Erbast members.
        If no empty cell is found, the member is removed from the population.
        '''
        for member in excess_members:
            nearby_cells = self.world.get_neighborhood_positions(current_pos[0], current_pos[1], 1)
            empty_cells = [cell for cell in nearby_cells if cell not in [m['position'] for m in self.creature_pop]]

            if empty_cells:
                new_pos = random.choice(empty_cells)
                if self.world.is_valid_move(*new_pos) and self.world.check_cell_limits(*new_pos):
                    member['position'] = new_pos
                    self.positions.add(new_pos)
            else:
                self.creature_pop.remove(member)
                if member['position'] in self.positions:
                    self.positions.remove(member['position'])

    def spawn(self):
        '''
        Spawns new Erbast upon death of the parent, distributing its attributes between the offspring.
        '''
        super().spawn(AGING_ERBAST, LIFETIME_ERBAST, MAX_HERD)

    def overwhelmed(self):
        '''
        Removes Erbast from the world when their cell is completely eaten by Carviz.
        '''
        super().overwhelmed()

class Carviz(Species):
    def __init__(self, world):
        '''
        Carviz initialization.
        - memory: strategic decisions are based on the last visited cells, the successful hunts, and dangerous cells.
        '''
        super().__init__(world)
        self.memory = {
            "last_visited": set(),
            "successful_hunts": set(),  
            "dangerous_cells": set()  
        }

    def place_randomly(self, count):
        '''
        Places Carviz randomly in the world, initializing their attributes.
        '''
        def create_carviz_data(self, x, y):
            return {
                'position': (x, y),
                'energy': random.randint(50, 100),  
                'lifetime': LIFETIME_CARVIZ,  
                'age': 0,  
                'social_attitude': random.uniform(0, 1),  
                'memory': {
                    "last_visited": set(),
                    "successful_hunts": set(),
                    "dangerous_cells": set()
                },
                'move_time': 0
            }
        
        super().place_randomly(count, create_carviz_data)
    
    def add_creature(self, creature_data):
        '''
        Adds Carviz to the creature population.
        '''
        self.creature_pop.append(creature_data)

    def move(self):
        '''
        Moves the Carviz in the world.
        '''
        super().move(self.evaluate_cell_with_memory, ENERGY_THRESHOLD_CARVIZ, SA_THRESHOLD_CARVIZ)

    def evaluate_cell_with_memory(self, pride_member, position):
        '''
        Evaluates the cell based on prey richness, hunting memory, and dangerous cells.
        '''
        prey_bonus = 5 if self.world.evaluate_cell(position[0], position[1], 'Erbast') > 1 else 0
        hunt_bonus = 2 if position in pride_member['memory']['successful_hunts'] else 0
        danger_penalty = 3 if position in pride_member['memory']['dangerous_cells'] else 0
        return max(0, prey_bonus + hunt_bonus - danger_penalty)

    def fuse_prides_or_fight(self):
        '''
        Fuses prides or lets them fight for dominance based on social attitudes.
        Prides are sorted by size so that the smallest have a chance to fuse before potential fights.
        Memoeries of the fused prides' members are combined, allowing for more strategic decisions.
        '''
        prides_by_pos = self.group_by_position(group_by_time=True)

        new_pop = []
        for pos, pride_time in prides_by_pos.items():
            prides_list = list(pride_time.values())

            prides_list.sort(key=len)

            while len(prides_list) > 1:
                pride_1 = prides_list.pop(0)
                pride_2 = prides_list.pop(0)

                avg_social_attitude_1 = sum(p['social_attitude'] for p in pride_1) / len(pride_1)
                avg_social_attitude_2 = sum(p['social_attitude'] for p in pride_2) / len(pride_2)

                if avg_social_attitude_1 > random.random() and avg_social_attitude_2 > random.random():
                    fused_pride = pride_1 + pride_2
                    combined_memory = {}
                    for member in fused_pride:
                        combined_memory.update(member['memory'])
                        member['social_attitude'] = sum([m['social_attitude'] for m in fused_pride]) / len(fused_pride)
                    for member in fused_pride:
                        member['memory'] = combined_memory
                    prides_list.append(fused_pride)
                else:
                    winning_pride = self.pride_fight(pride_1, pride_2)
                    prides_list.append(winning_pride)

                prides_list.sort(key=len)

            if prides_list:
                cell_new_pop = prides_list[0][:MAX_PRIDE]
                new_pop.extend(cell_new_pop)

        self.creature_pop = new_pop
        self.positions = set(member['position'] for member in self.creature_pop)

    def pride_fight(self, pride_1, pride_2):
        '''
        A last-blood match between two prides. The pride led by the strongest Carviz wins.
        The loser is removed from the population, the winner loses 10% of its energy.
        The winning pride gains a boost in social attitude and its members' memories are updated with the current position.
        '''
        while pride_1 and pride_2:
            champion_1 = max(pride_1, key=lambda p: p['energy'])
            champion_2 = max(pride_2, key=lambda p: p['energy'])
            
            if champion_1['energy'] > champion_2['energy']:
                loser = pride_2.pop(pride_2.index(champion_2))
                champion_1['energy'] *= 0.9 
                champion_1['energy'] = max(champion_1['energy'], 0)
            else:
                loser = pride_1.pop(pride_1.index(champion_1))
                champion_2['energy'] *= 0.9 
                champion_2['energy'] = max(champion_2['energy'], 0)
            if loser['position'] in self.positions:
                self.positions.remove(loser['position'])
        
        winning_pride = pride_1 if pride_1 else pride_2
        for member in winning_pride:
            member['social_attitude'] += 0.2  
            member['social_attitude'] = min(1, member['social_attitude'])  

        if winning_pride:
            position = winning_pride[0]['position']
            for member in winning_pride:
                if 'memory' in member and 'dangerous_cells' in member['memory']:  
                    member['memory']['dangerous_cells'].add(position)

        return winning_pride

    def pride_hunt(self, erbast):
        '''
        Carviz hunt Erbast in cells with one remaining pride.
        The strongest Erbast is hunted and its energy is distributed among the pride members.
        Chance of success is based on the ratio of pride energy to the hunted Erbast's energy, but is capped at 70%.
        If successful, the pride's attitude is boosted and the Erbast's energy is shared among the pride members.
        If not, the pride's attitude is decreased and the Erbast's memory is updated with the current position.
        '''
        pride_by_pos = self.group_by_position() 
        
        for pos, pride in pride_by_pos.items():
            erbast_in_cell = [herd_member for herd_member in self.world.erbast.creature_pop if herd_member['position'] == pos]
            
            if erbast_in_cell:
                strongest_erbast = max(erbast_in_cell, key=lambda e: e['energy'])
                success_probability = min(0.7, sum([p['energy'] for p in pride]) / (sum([p['energy'] for p in pride]) + strongest_erbast['energy']))
                
                if random.random() < success_probability:
                    energy_gain = strongest_erbast['energy'] * 0.8  
                    erbast_in_cell.remove(strongest_erbast)
                    
                    self.world.erbast.creature_pop.remove(strongest_erbast)
                    if strongest_erbast['position'] in self.world.erbast.positions:
                        self.world.erbast.positions.remove(strongest_erbast['position'])

                    for carviz in pride:
                        carviz['energy'] += int(energy_gain // len(pride))
                        carviz['energy'] = min(carviz['energy'], 100)
                    
                        carviz['social_attitude'] = min(1, carviz['social_attitude'] + 0.1)

                        if pos not in carviz['memory']['successful_hunts']:
                            carviz['memory']['successful_hunts'].add(pos)
                    
                else:
                    for carviz in pride:
                        carviz['social_attitude'] = max(0, carviz['social_attitude'] - 0.1)

                for erbast in erbast_in_cell:
                    if pos not in erbast['memory']['dangerous_cells']:
                        erbast['memory']['dangerous_cells'].add(pos)

            else:
                for carviz in pride:
                    carviz['social_attitude'] = max(0, carviz['social_attitude'] - 0.1)
            
    def spawn(self):
        '''
        Spawns new Carviz upon death of the parent, distributing its attributes between the offspring.
        '''
        super().spawn(AGING_CARVIZ, LIFETIME_CARVIZ, MAX_PRIDE)

    def overwhelmed(self):
        '''
        Removes Carviz from the world when their cell is completely eaten by Erbast.
        '''
        super().overwhelmed()


# GUI classes
class PlanisussGUI:
    def __init__(self, world):
        '''
        Initializes the main GUI for the simulation.
        Sets up the simulation parameters, data, and GUI elements.
        Images are loaded and resized for a nicer visualization of zoomed cells.
        '''
        self.world = world

        self.paused = False
        self.terminated = False
        self.pause_button = None 
        self.speed = 100  
        self.speed_factor = 1.5  
        self.current_frame = 0
        self.anim = None
        self.anim_running = False
        self.last_update_time = time.time()
        self.graph_window = None
        self.parameter_window = None

        self.population_data = {'time': [], 'erbast': [], 'carviz': []}
        self.vegetob_density_data = {'time': [], 'density': []}
        self.energy_levels_data = {'time': [], 'erbast_energy': [], 'carviz_energy': []}
        self.last_time = 0

        self.simulation_params = {
            "Simulation length": NUMDAYS,
            "Neighborhood": NEIGHBORHOOD,
            "Max pride size": MAX_PRIDE,
            "Max herd size": MAX_HERD,
            "Erbast aging": AGING_ERBAST,
            "Carviz aging": AGING_CARVIZ,
            "Vegetob growth": VEGETOB_GROWTH,
            "Erbast lifetime": LIFETIME_ERBAST,
            "Carviz lifetime": LIFETIME_CARVIZ,
            "Erbast energy threshold": ENERGY_THRESHOLD_ERBAST,
            "Carviz energy threshold": ENERGY_THRESHOLD_CARVIZ,
            "Erbast social attitude threshold": SA_THRESHOLD_ERBAST,
            "Carviz social attitude threshold": SA_THRESHOLD_CARVIZ
        }
        
        self.root = tk.Tk()
        self.root.title("Planisuss World")
        self.root.geometry("1400x1000")
        self.root.configure(bg='deepskyblue')  

        self.font = tkFont.Font(family="Fixedsys", size=24)

        self.erbast_image = Image.open("erbast.png")
        self.carviz_image = Image.open("carviz.png")
        self.water_bg = Image.open("water.png")
        self.ground_bg = Image.open("vegetob.png")

        self.erbast_image = self.erbast_image.resize((80, 80), Image.LANCZOS)
        self.carviz_image = self.carviz_image.resize((80, 80), Image.LANCZOS)
        self.water_bg = self.water_bg.resize((300, 300), Image.LANCZOS)
        self.ground_bg = self.ground_bg.resize((300, 300), Image.LANCZOS)

        self.setup_gui()

    def setup_gui(self):
        '''
        Sets up the GUI elements, including buttons and the world display.
        The left frame contains the buttons, and the right frame contains the world display.
        The buttons allow for control of the simulation, such as pausing, terminating, restarting,
        speeding up, slowing down, displaying graphs, and changing parameters.
        '''
        # Title
        title_font = tkFont.Font(family="Fixedsys", size=50, weight="bold")
        title = tk.Label(self.root, text="Planisuss World", font=title_font, bg='deepskyblue', fg='navy')
        title.pack(pady=20)

        # Main frame
        main_frame = tk.Frame(self.root, bg='deepskyblue')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for buttons
        left_frame = tk.Frame(main_frame, width=400, bg='deepskyblue')
        left_frame.pack(side=tk.LEFT, padx=50, pady=0, fill=tk.BOTH, expand=True)
        left_frame.pack_propagate(False)  

        # Right frame for map
        right_frame = tk.Frame(main_frame, bg='deepskyblue')
        right_frame.pack(side=tk.RIGHT, padx=0, pady=0, fill=tk.BOTH, expand=True)

        # Buttons
        buttons = [
            ("Pause", self.pause),
            ("Terminate", self.terminate),
            ("Restart", self.restart),  
            ("Speed Up", self.speed_up),
            ("Slow Down", self.slow_down),
            ("Display Graphs", self.display_graphs),
            ("Change Parameters", self.change_parameters) 
        ]

        # Top spacer frame to center buttons
        top_spacer = tk.Frame(left_frame, bg='deepskyblue')
        top_spacer.pack(expand=True, fill=tk.Y)

        # Button frame
        button_frame = tk.Frame(left_frame, bg='deepskyblue')
        button_frame.pack(expand=True)

        for text, command in buttons:
            button_width = self.font.measure(text) // self.font.measure('0') + 2 
            button = tk.Button(button_frame, text=text, command=command,
                               font=self.font, bg='saddlebrown', fg='white', activebackground='brown',
                               activeforeground='white', width=button_width)
            button.pack(pady=15, anchor='center')

            if text == "Pause":
                self.pause_button = button  

        # Bottom spacer frame to center buttons
        bottom_spacer = tk.Frame(left_frame, bg='deepskyblue')
        bottom_spacer.pack(expand=True, fill=tk.Y)

        # Creating and displaying the world
        self.fig, self.ax, self.im, self.erbast_scatter, self.carviz_scatter = display_world(self.world)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect('button_press_event', self.on_cell_click)

        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        self.anim = animation.FuncAnimation(self.fig, self.update, frames=NUMDAYS, interval=self.speed, blit=False, repeat=False)

    def pause(self):
        '''
        Pauses or resumes the simulation.
        The pause button will display "Resume" if the simulation is paused, and "Pause" if it is running.
        '''
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="Resume")
            print("Simulation paused")
            if self.anim and self.anim.event_source:
                self.anim.event_source.stop()
        else:
            self.pause_button.config(text="Pause")
            print("Simulation resumed")
            if self.anim and self.anim.event_source:
                self.anim.event_source.start()
            elif not self.anim.event_source:
                print("Event source not found - pause")  # Not useful for the simulation, but a debug print statement 
                                                         # to check if the event source is found
        
    def terminate(self):
        '''
        Terminates the simulation. A dialog box will ask for confirmation.
        '''
        dialog = CustomDialog(self.root, "Terminate Simulation", "Are you sure you want to terminate the simulation?")
        if dialog.result:
            self.terminated = True
            self.running = False
            if self.anim and hasattr(self.anim, 'event_source'):
                print("Stopping animation")
                try:
                    self.anim.event_source.stop()
                except Exception as e:
                    print(f"Error stopping animation: {e}")  # Again, not useful for the simulation, but for debugging
            self.anim = None
            self.anim_running = False
            self.root.quit()
            print("Simulation terminated")
        else:
            print("Termination cancelled")

    def restart(self):
        '''
        Restarts the simulation. A dialog box will ask for confirmation.
        The reset_simulation() function is called to reset the simulation parameters and the world.
        '''
        dialog = CustomDialog(self.root, "Restart Simulation", "Are you sure you want to restart the simulation?")
        if dialog.result:
            print("Restarting simulation")
            self.reset_simulation()
        else:
            print("Restart cancelled")

    def reset_simulation(self):
        '''
        Resets the simulation. The world is reset to its initial state and the simulation parameters are updated.
        '''
        self.world = World(size=NUMCELLS, ground_ratio=0.8)
        self.reset_simulation_params()
        self.last_update_time = time.time()
    
    def speed_up(self):
        '''
        Increases the speed of the simulation by a factor of 1.5. 
        The speed cannot be increased beyond the maximum speed, which is 1000 ms.
        '''
        if self.speed <= 1:
            print("Speed already at maximum")
        else:
            new_speed = max(1, math.floor(self.speed / self.speed_factor))
            self.speed = new_speed
            print(f"Speed increased. New interval: {self.speed} ms")
            self.anim.event_source.interval = self.speed
            if not self.anim.event_source:
                print("Event source not found - speed up")  # Not useful for the simulation, but for debugging

    def slow_down(self):
        '''
        Decreases the speed of the simulation by a factor of 1.5.
        The speed cannot be decreased beyond the minimum speed, which is 1 ms.
        Math.ceil is used to ensure the speed doesn't get stuck at its minimum value.
        '''
        if self.speed >= 1000:
            print("Speed already at minimum")
        else:
            new_speed = min(1000, math.ceil(self.speed * self.speed_factor))
            self.speed = new_speed
            print(f"Speed decreased. New interval: {self.speed} ms")
            self.anim.event_source.interval = self.speed
            if not self.anim.event_source:
                print("Event source not found - slow down")  # Not useful for the simulation, but for debugging

    def display_graphs(self):
        '''
        Displays the window for selecting the graph to visualize.
        '''
        self.graph_window = None
        if self.graph_window is None:
            self.graph_window = GraphSelectionGUI(self.root, self.on_graph_selected)

    def on_graph_selected(self, graph):
        '''
        Displays the selected graph.
        '''
        print(f"Selected graph: {graph}")
        if graph == "Erbast and Carviz Population Over Time":
            self.show_population_graph()
        elif graph == "Vegetob Density Over Time":
            self.show_vegetob_density_graph()
        elif graph == "Energy Levels Over Time":
            self.show_energy_levels_graph()

    def update_population_data(self):
        '''
        Updates the population data, that is, the numerosity of Erbast and Carviz over time.
        '''
        self.population_data['time'].append(self.last_time + self.current_frame)
        self.population_data['erbast'].append(len(self.world.erbast.creature_pop))
        self.population_data['carviz'].append(len(self.world.carviz.creature_pop))

    def show_population_graph(self):
        '''
        Displays the population graph.
        '''
        print("Creating population graph...")
        self.graph_window = GraphWindow(
            self.root, 
            "Erbast and Carviz Population Over Time", 
            self.population_data,
            "Time",
            "Population Size",
            ["Erbast", "Carviz"],
            ["yellow", "red"]
            )
        self.graph_window.animate()
        print("Population graph displayed")

    def update_vegetob_density_data(self):
        '''
        Updates the vegetob density data, that is, the average density of Vegetob over time.
        '''
        self.vegetob_density_data['time'].append(self.last_time + self.current_frame)

        densities = list(self.world.vegetob.intensity.values())
        if densities:  
            average_density = sum(densities) / len(densities)
        else:
            average_density = 0

        self.vegetob_density_data['density'].append(average_density)

    def show_vegetob_density_graph(self):
        '''
        Displays the vegetob density graph.
        '''
        print("Creating vegetob density graph...")
        self.graph_window = GraphWindow(
            self.root, 
            "Vegetob Density Over Time", 
            self.vegetob_density_data,
            "Time",
            "Vegetob Density",
            ["Vegetob Density"],
            ["green"]
            )
        self.graph_window.animate()
        print("Vegetob density graph displayed")

    def update_energy_levels_data(self):
        '''
        Updates the energy levels data, that is, the average energy levels of Erbast and Carviz over time.
        '''
        self.energy_levels_data['time'].append(self.last_time + self.current_frame)

        erbast_energies = [erbast['energy'] for erbast in self.world.erbast.creature_pop]
        carviz_energies = [carviz['energy'] for carviz in self.world.carviz.creature_pop]

        if erbast_energies:
            average_erbast_energy = sum(erbast_energies) / len(erbast_energies)
        else:
            average_erbast_energy = 0 

        if carviz_energies:
            average_carviz_energy = sum(carviz_energies) / len(carviz_energies)
        else:
            average_carviz_energy = 0 

        self.energy_levels_data['erbast_energy'].append(average_erbast_energy)
        self.energy_levels_data['carviz_energy'].append(average_carviz_energy)

    def show_energy_levels_graph(self):
        '''
        Displays the energy levels graph.
        '''
        print("Creating energy levels graph...")
        self.graph_window = GraphWindow(
            self.root, 
            "Energy Levels Over Time", 
            self.energy_levels_data,
            "Time",
            "Energy Levels",
            ["Erbast_Energy", "Carviz_Energy"],
            ["yellow", "red"]
            )
        self.graph_window.animate()
        print("Energy levels graph displayed")

    def change_parameters(self):
        '''
        Temporarily stops the simulation for parameter change in real time.
        '''
        if self.anim and self.anim.event_source and not self.paused:
            self.anim.event_source.stop()
        self.last_time += self.current_frame
        self.current_frame = 0
        self.parameter_window = ParameterChangeGUI(self, self.root, self.simulation_params)
        if not self.anim.event_source:
            print("Event source not found - params")  # Not useful for the simulation, but for debugging

    def reset_simulation_params(self):
        '''
        Resets the simulation parameters to the new parameters. 
        The changes are applied upon clicking the "Apply" button in the parameter change GUI.
        '''
        self.running = False
        if self.anim:
            self.anim.event_source.stop()
        self.anim = None

        global VEGETOB_GROWTH, MAX_HERD, MAX_PRIDE, AGING_ERBAST, AGING_CARVIZ, LIFETIME_ERBAST, LIFETIME_CARVIZ, NEIGHBORHOOD
        global ENERGY_THRESHOLD_ERBAST, ENERGY_THRESHOLD_CARVIZ, SA_THRESHOLD_ERBAST, SA_THRESHOLD_CARVIZ
        
        VEGETOB_GROWTH = self.simulation_params["Vegetob growth"]
        MAX_HERD = self.simulation_params["Max herd size"]
        MAX_PRIDE = self.simulation_params["Max pride size"]
        AGING_ERBAST = self.simulation_params["Erbast aging"]
        AGING_CARVIZ = self.simulation_params["Carviz aging"]
        LIFETIME_ERBAST = self.simulation_params["Erbast lifetime"]
        LIFETIME_CARVIZ = self.simulation_params["Carviz lifetime"]
        NEIGHBORHOOD = self.simulation_params["Neighborhood"]
        ENERGY_THRESHOLD_ERBAST = self.simulation_params["Erbast energy threshold"]
        ENERGY_THRESHOLD_CARVIZ = self.simulation_params["Carviz energy threshold"]
        SA_THRESHOLD_ERBAST = self.simulation_params["Erbast social attitude threshold"]
        SA_THRESHOLD_CARVIZ = self.simulation_params["Carviz social attitude threshold"]

        self.ax.clear()

        new_fig, new_ax, new_im, new_erbast_scatter, new_carviz_scatter = display_world(self.world)
        
        self.anim = animation.FuncAnimation(self.fig, self.update, frames=self.simulation_params["Simulation length"], 
                                            interval=self.speed, blit=False, repeat=False)
        
        # Update the existing figure and axes with the new content
        self.im = self.ax.imshow(new_im.get_array(), origin='lower')
        self.erbast_scatter = self.ax.scatter([], [], c='yellow', s=[], zorder=2)
        self.carviz_scatter = self.ax.scatter([], [], c='red', s=[], zorder=3)

        self.ax.set_xlim(0, self.world.size)
        self.ax.set_ylim(0, self.world.size)
        self.ax.set_aspect('equal')

        self.ax.set_axis_off()

        plt.close(new_fig)

        self.canvas.draw()  
        self.anim_running = True
        self.paused = False
        self.terminated = False
        self.running = True
        if self.pause_button:
            self.pause_button.config(text="Pause")

        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        print("Simulation reset with new parameters")

    def on_cell_click(self, event):
        '''
        Displays the cell zoom window when the simulation is paused and a cell is clicked.
        Event.inaxes is used to check if the click is within the axes of the plot.
        '''
        if self.paused and event.inaxes == self.ax:
            col = int(event.xdata)
            row = int(event.ydata)
            if 0 <= col < self.world.size and 0 <= row < self.world.size:
                self.show_cell_zoom(row, col)

    def show_cell_zoom(self, row, col):
        '''
        Shows a zoomed view of the selected cell. 
        Both background material and creatures are represented by images of my own design, which I implemented
        to make the GUI more appealing and more informative.
        The vegetob intensity is represented by a semi-transparent green overlay.

        Below the zoomed view, a text field displays the cell type, vegetob intensity, number of Erbast and Carviz,
        and the details of the Erbast and Carviz in the cell.
        '''
        zoom_window = tk.Toplevel(self.root)
        zoom_window.title(f"Cell Zoom ({row}, {col})")
        zoom_window.geometry("700x1000")
        zoom_window.configure(bg='deepskyblue')

        font = tkFont.Font(family="Fixedsys", size=20)

        main_frame = tk.Frame(zoom_window, bg='deepskyblue', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame, width=300, height=300, bg='white')
        canvas.pack(pady=10)

        canvas.images = []
        cell_type = "Water" if self.world.grid[row, col] == 0 else "Ground"
        if cell_type == "Water":
            bg_image = ImageTk.PhotoImage(self.water_bg)
        else:
            bg_image = ImageTk.PhotoImage(self.ground_bg)

        canvas.create_image(150, 150, image=bg_image)
        canvas.bg_image = bg_image

        if cell_type == "Ground":
            # Applying a semi-transparent green overlay based on Vegetob intensity
            vegetob_intensity = self.world.vegetob.intensity.get((row, col), 0)
            green_value = int(255 * (1 - vegetob_intensity / 100))
            red_value = int(green_value * 0.5)
            blue_value = int(green_value * 0.5)
            overlay = Image.new('RGBA', (300, 300), (red_value, green_value, blue_value, 128))  
            overlay_photo = ImageTk.PhotoImage(overlay)
            canvas.create_image(150, 150, image=overlay_photo)
            canvas.overlay_photo = overlay_photo  

            erbast_list = [e for e in self.world.erbast.creature_pop if e['position'] == (row, col)]
            carviz_list = [c for c in self.world.carviz.creature_pop if c['position'] == (row, col)]

            all_creatures = erbast_list + carviz_list
            random.shuffle(all_creatures)

            # Function checking if a position is valid (not overlapping with existing creatures if possible)
            def is_valid_position(x, y, placed_creatures):
                for px, py in placed_creatures:
                    if math.sqrt((x - px)**2 + (y - py)**2) < self.erbast_image.width:
                        return False
                return True

            placed_creatures = []

            for creature in all_creatures:
                attempts = 0

                # Up to 50 attempts to find a non-overlapping position for each creature
                while attempts < 50:  
                    x = random.randint(self.erbast_image.width//2, 300 - self.erbast_image.width//2)
                    y = random.randint(self.erbast_image.height//2, 300 - self.erbast_image.height//2)
                    if is_valid_position(x, y, placed_creatures):
                        if creature in erbast_list:
                            photo = ImageTk.PhotoImage(self.erbast_image)
                        elif creature in carviz_list:
                            photo = ImageTk.PhotoImage(self.carviz_image)
                        else:
                            break  

                        canvas.create_image(x, y, image=photo, anchor=tk.CENTER)
                        canvas.images.append(photo)
                        placed_creatures.append((x, y))
                        break
                    attempts += 1

                if attempts == 50:
                    # If a non-overlapping position is not found after 50 attempts,
                    # just place the creature at the last attempted position
                    if creature in erbast_list:
                        photo = ImageTk.PhotoImage(self.erbast_image)
                    elif creature in carviz_list:
                        photo = ImageTk.PhotoImage(self.carviz_image)
                    else:
                        continue 

                    canvas.create_image(x, y, image=photo, anchor=tk.CENTER)
                    canvas.images.append(photo)
                    placed_creatures.append((x, y))


        # Adding cell details
        tk.Label(main_frame, text=f"Cell Type: {cell_type}", font=font, bg='deepskyblue', fg='navy').pack(anchor='w')
        if cell_type == "Ground":
            tk.Label(main_frame, text=f"Vegetob Intensity: {vegetob_intensity}", font=font, bg='deepskyblue', fg='navy').pack(anchor='w')
            tk.Label(main_frame, text=f"Number of Erbast: {len(erbast_list)}", font=font, bg='deepskyblue', fg='navy').pack(anchor='w')

            if erbast_list:
                details_frame = tk.Frame(main_frame, bg='deepskyblue')
                details_frame.pack(fill=tk.X, pady=10)
                text_height = max(3, min(10, int(len(erbast_list)*1.5)))
                details_text = tk.Text(details_frame, height=text_height, width=50, font=font, bg='saddlebrown', fg='white')
                details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                details_scroll = tk.Scrollbar(details_frame, command=details_text.yview)
                details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                details_text.config(yscrollcommand=details_scroll.set)

                for i, erbast in enumerate(erbast_list):
                    details_text.insert(tk.END, f"Erbast {i+1}: Energy: {erbast['energy']}, Age: {erbast['age']}, SA: {erbast['social_attitude']:.2f}\n")
                
            tk.Label(main_frame, text=f"Number of Carviz: {len(carviz_list)}", font=font, bg='deepskyblue', fg='navy').pack(anchor='w')

            if carviz_list:
                details_frame = tk.Frame(main_frame, bg='deepskyblue')
                details_frame.pack(fill=tk.X, pady=10)
                text_height = max(3, min(10, int(len(carviz_list)*1.5)))
                details_text = tk.Text(details_frame, height=text_height, width=50, font=font, bg='saddlebrown', fg='white')
                details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                details_scroll = tk.Scrollbar(details_frame, command=details_text.yview)
                details_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                details_text.config(yscrollcommand=details_scroll.set)
                
                for i, carviz in enumerate(carviz_list):
                    details_text.insert(tk.END, f"Carviz {i+1}: Energy: {carviz['energy']}, Age: {carviz['age']}, SA: {carviz['social_attitude']:.2f}\n")
        
        close_button = tk.Button(main_frame, text="Close", command=zoom_window.destroy,
                                font=font, bg='saddlebrown', fg='white',
                                activebackground='brown', activeforeground='white')
        close_button.pack(pady=10)

    def update(self, frame):
        '''
        Updates the simulation by one frame.
        The update is done by checking the time elapsed since the last update and comparing it to the speed of the simulation.
        The world, data for graphs and images are updated accordingly.
        If the simulation reaches its maximum length (NUMDAYS), the animation is stopped.
        '''
        current_time = time.time()
        if not self.paused and not self.terminated and current_time - self.last_update_time >= self.speed / 1000:   
            if self.current_frame < self.simulation_params["Simulation length"]:
                self.world.update()
                self.update_population_data()
                self.update_vegetob_density_data()
                self.update_energy_levels_data()

                #Updating Vegetob
                image = create_world_image(self.world)
                self.im.set_array(image)

                #Updating Erbast 
                erbast_positions = np.array([erbast['position'] for erbast in self.world.erbast.creature_pop])
                erbast_sizes = np.array([self.world.erbast_density(pos) * 10 for pos in erbast_positions])
                if len(erbast_positions) > 0:
                    self.erbast_scatter.set_offsets(erbast_positions[:, [1, 0]])
                    self.erbast_scatter.set_sizes(erbast_sizes)
                else:
                    self.erbast_scatter.set_offsets(np.empty((0, 2)))
                    self.erbast_scatter.set_sizes([])

                #Updating Carviz
                carviz_positions = np.array([carviz['position'] for carviz in self.world.carviz.creature_pop])
                carviz_sizes = np.array([self.world.carviz_density(pos) * 10 for pos in carviz_positions])
                if len(carviz_positions) > 0:
                    self.carviz_scatter.set_offsets(carviz_positions[:, [1, 0]])
                    self.carviz_scatter.set_sizes(carviz_sizes)
                else:
                    self.carviz_scatter.set_offsets(np.empty((0, 2)))
                    self.carviz_scatter.set_sizes([])
                
                self.canvas.draw()
                self.current_frame += 1
                self.last_update_time = current_time  
            else:
                print("Simulation completed")
                self.anim.event_source.stop()
                self.last_time += self.current_frame

    def run(self):
        '''
        Starts the simulation and handles the window close event, in case the user wants to quit the simulation
        without using the "terminate" button.
        '''
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)    
        self.root.mainloop()

    def on_closing(self):
        '''
        Displays a dialog asking the user if they want to quit the simulation.
        If the user clicks "Yes", the simulation is terminated and the window is closed.
        If the user clicks "No", the dialog is closed and the simulation continues.
        '''
        dialog = CustomDialog(self.root, "Quit", "Do you want to quit the simulation?")
        if dialog.result:
            self.terminated = True
            self.root.quit()  

class GraphSelectionGUI:
    def __init__(self, parent, on_graph_selected):
        '''
        Initializes the GUI to select the graph to visualize.
        '''
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Select Graph")
        self.window.geometry("800x300")
        self.window.configure(bg='deepskyblue')

        self.on_graph_selected = on_graph_selected

        font = tkFont.Font(family="Fixedsys", size=18)

        label = tk.Label(self.window, text="What graph would you like to visualize?", font=font, bg='deepskyblue', fg='navy')
        label.pack(pady=20)

        button_frame = tk.Frame(self.window, bg='deepskyblue')
        button_frame.pack(pady=10)

        graphs = [
            "Erbast and Carviz Population Over Time", 
            "Vegetob Density Over Time", 
            "Energy Levels Over Time"
            ]  

        for graph in graphs:
            button = tk.Button(button_frame, text=graph, command=lambda g=graph: self.select_graph(g),
                               font=font, bg='saddlebrown', fg='white', activebackground='brown', activeforeground='white')
            button.pack(pady=5)

    def select_graph(self, graph):
        '''
        Selects the graph to visualize and closes the window.
        '''
        self.on_graph_selected(graph)
        self.window.destroy()

class GraphWindow:
    def __init__(self, parent, title, data, x_label, y_label, line_labels, colors):
        '''
        Initializes the window to display the selected graph.
        '''
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry("1000x500")
        self.window.configure(bg='deepskyblue')

        self.data = data
        self.x_label = x_label
        self.y_label = y_label

        font = tkFont.Font(family="Fixedsys", size=30, weight="bold")

        title_label = tk.Label(self.window, text=title, font=font, bg='deepskyblue', fg='navy')
        title_label.pack(pady=5)

        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor('deepskyblue')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('deepskyblue')
        self.ax.set_xlabel(x_label, family='Courier New', fontsize=16, weight="bold", color='navy')
        self.ax.set_ylabel(y_label, family='Courier New', fontsize=16, weight="bold", color='navy')
        self.line_labels = line_labels
        self.colors = colors
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.update_graph()

    def update_graph(self):
        '''
        Updates the graph with new data.
        '''
        self.ax.clear()
        for i, label in enumerate(self.line_labels):

            label1 = label.split()[0].lower()
            if label1 == "vegetob":
                label1 = "density"

            self.ax.plot(self.data['time'], self.data[label1], label=label, color=self.colors[i], linewidth=2)

        self.ax.set_xlabel(self.x_label, family='Courier New', fontsize=16, weight="bold", color='navy')
        self.ax.set_ylabel(self.y_label, family='Courier New', fontsize=16, weight="bold", color='navy')

        self.ax.legend()
        self.canvas.draw()

    def animate(self):
        '''
        Updates the graph every second, allowing for real-time visualization of the data.
        '''
        self.update_graph()
        self.window.after(1000, self.animate)

class CustomDialog(tk.Toplevel):
    def __init__(self, parent, title, message):
        '''
        Initializes a custom dialog for user input.
        '''
        tk.Toplevel.__init__(self, parent)
        self.title(title)
        self.result = None
        self.configure(bg='deepskyblue')

        font = tkFont.Font(family="Fixedsys", size=18)

        tk.Label(self, text=message, font=font, bg='deepskyblue', fg='navy').pack(padx=20, pady=20)

        button_frame = tk.Frame(self, bg='deepskyblue')
        button_frame.pack(pady=10)

        yes_button = tk.Button(button_frame, text="Yes", command=self.yes, 
                               font=font, bg='saddlebrown', fg='white',
                               activebackground='brown', activeforeground='white')
        yes_button.pack(side=tk.LEFT, padx=10)

        no_button = tk.Button(button_frame, text="No", command=self.no, 
                              font=font, bg='saddlebrown', fg='white',
                              activebackground='brown', activeforeground='white')
        no_button.pack(side=tk.LEFT, padx=10)

        self.protocol("WM_DELETE_WINDOW", self.no)
        self.transient(parent)
        self.grab_set()
        self.center_window()
        parent.wait_window(self)

    def center_window(self):
        '''
        Centers the dialog window on the screen.
        '''
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def yes(self):
        '''
        Sets the result to True and destroys the dialog.
        '''
        self.result = True
        self.destroy()

    def no(self):
        '''
        Sets the result to False and destroys the dialog.
        '''
        self.result = False
        self.destroy()

class ParameterChangeGUI:
    def __init__(self, GUI, parent, current_params):
        '''
        Initializes the GUI to change simulation parameters.
        Parameters are displayed as sliders and can be adjusted in real time.
        '''
        self.parent = parent
        self.gui = GUI
        self.window = tk.Toplevel(parent)
        self.window.title("Change Simulation Parameters")
        self.window.geometry("900x950")
        self.window.configure(bg='deepskyblue')

        self.current_params = current_params
        self.new_params = {}

        self.param_ranges = {
            "Neighborhood": (1, 5),
            "Max herd size": (1, 500),
            "Max pride size": (1, 500),
            "Erbast aging": (1, 20),
            "Carviz aging": (1, 20),
            "Vegetob growth": (1, 10),
            "Erbast lifetime": (1, 100),
            "Carviz lifetime": (1, 100), 
            "Erbast energy threshold": (1, 100),
            "Carviz energy threshold": (1, 100),
            "Erbast social attitude threshold": (0, 1),
            "Carviz social attitude threshold": (0, 1)
        }

        self.create_widgets()

    def create_widgets(self):
        '''
        Creates the sliders for each parameter.
        Social attitude thresholds for movement are displayed with a custom scale for more precision, 
        that is, including decimal points. Other parameters use an integer scale.
        '''
        font = tkFont.Font(family="Fixedsys", size=20, weight="bold")

        main_frame = tk.Frame(self.window, bg='deepskyblue', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        parameters = list(self.param_ranges.keys())

        for i, param in enumerate(parameters):
            frame = tk.Frame(main_frame, bg='deepskyblue')
            frame.pack(fill=tk.X, pady=5)

            label = tk.Label(frame, text=f"{param}:", font=font, bg='deepskyblue', fg='navy')
            label.pack(side=tk.LEFT)

            min_val, max_val = self.param_ranges[param]

            if "social attitude threshold" in param:
                slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                                  length=400, font=font, bg='saddlebrown', fg='white',
                                  troughcolor='darkgreen', activebackground='brown', 
                                  resolution=0.01, digits=2)
                current_value = self.current_params.get(param, (min_val + max_val) / 2)
                slider.set(current_value)
                slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

                self.new_params[param] = slider
            else:
                slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                                  length=400, font=font, bg='saddlebrown', fg='white',
                                  troughcolor='darkgreen', activebackground='brown')
                slider.set(self.current_params.get(param, (min_val + max_val) // 2))
                slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=20)
                
                self.new_params[param] = slider

        button_frame = tk.Frame(main_frame, bg='deepskyblue')
        button_frame.pack(pady=20)

        apply_button = tk.Button(button_frame, text="Apply", command=self.apply_changes,
                                 font=font, bg='saddlebrown', fg='white',
                                 activebackground='brown', activeforeground='white')
        apply_button.pack(side=tk.LEFT, padx=10)

        cancel_button = tk.Button(button_frame, text="Cancel", command=self.on_cancel,
                                  font=font, bg='saddlebrown', fg='white',
                                  activebackground='brown', activeforeground='white')
        cancel_button.pack(side=tk.LEFT, padx=10)
        
    def apply_changes(self):
        '''
        Updates the parameters with new values, resumes the simulation and closes the window.
        '''
        for param, slider in self.new_params.items():
            if "social attitude threshold" in param:
                value = round(float(slider.get()), 2)
            else:
                value = slider.get()
                if param in ["Neighborhood", "Max pride size", "Max herd size", "Erbast aging", "Carviz aging", "Vegetob growth",  
                             "Erbast lifetime", "Carviz lifetime", "Erbast energy threshold", "Carviz energy threshold"]:
                    value = int(value)
            self.current_params[param] = value
        
        print(f"Updated parameters: {self.current_params}")
        self.gui.reset_simulation_params()
        self.window.destroy()

    def on_cancel(self):
        '''
        Resumes the simulation and closes the window if the user changes their mind.
        '''
        self.gui.anim.event_source.start()
        self.window.destroy()


# Visualization functions
def create_world_image(world):
    '''
    Creates the world image, which is a numpy array of the world's grid.
    Water cells are displayed as blue, ground cells are displayed as a gradient of green based on Vegetob intensity.
    At the beginning of the simulation, ground cells with no Vegetob are displayed in a light green-brown color.
    '''
    image = np.zeros((world.size, world.size, 3))

    image[world.grid == 0] = colors.to_rgb('deepskyblue')

    min_vegetob_color = np.array(colors.to_rgb('darkkhaki'))  
    max_vegetob_color = np.array(colors.to_rgb('darkgreen')) 

    image[world.grid == 1] = min_vegetob_color

    for (i, j), intensity in world.vegetob.intensity.items():
        if world.grid[i, j] == 1:
            green_intensity = intensity / 100
            cell_color = min_vegetob_color * (1 - green_intensity) + max_vegetob_color * green_intensity
            image[i, j] = cell_color

    return image

def display_world(world):
    '''
    Displays the world with Erbast and Carviz, represented by yellow and red dots, whose size is proportional to their density.
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()

    image = create_world_image(world)
    
    im = ax.imshow(image, origin='lower')
    
    erbast_positions = np.array([erbast['position'] for erbast in world.erbast.creature_pop])
    carviz_positions = np.array([carviz['position'] for carviz in world.carviz.creature_pop])

    erbast_sizes = np.array([world.erbast_density(pos) * 10 for pos in erbast_positions])
    carviz_sizes = np.array([world.carviz_density(pos) * 10 for pos in carviz_positions])

    # Handling unidimensional cases
    if erbast_positions.ndim == 1:
        erbast_scatter = ax.scatter(erbast_positions[1], erbast_positions[0], c='yellow', s=erbast_sizes, zorder=2)
    else:
        erbast_scatter = ax.scatter(erbast_positions[:, 1], erbast_positions[:, 0], c='yellow', s=erbast_sizes, zorder=2)

    if carviz_positions.ndim == 1:
        carviz_scatter = ax.scatter(carviz_positions[1], carviz_positions[0], c='red', s=carviz_sizes, zorder=3)
    else:
        carviz_scatter = ax.scatter(carviz_positions[:, 1], carviz_positions[:, 0], c='red', s=carviz_sizes, zorder=3)

    plt.xlim(0, world.size)
    plt.ylim(0, world.size)
    ax.set_aspect('equal')

    fig.patch.set_facecolor('deepskyblue')  
    ax.set_facecolor('deepskyblue')  
    
    return fig, ax, im, erbast_scatter, carviz_scatter

def update_world(frame, world, ax, im, erbast_scatter, carviz_scatter):
    '''
    Calls the world update method and updates the world image accordingly.
    Erbast and Carviz dots increase or decrease in size based on the updated density.
    '''
    world.update()

    image = create_world_image(world)
    im.set_array(image)

    erbast_positions = np.array([erbast['position'] for erbast in world.erbast.creature_pop])
    carviz_positions = np.array([carviz['position'] for carviz in world.carviz.creature_pop])

    # Handling Erbast updates for unidimensional and multidimensional arrays
    if len(erbast_positions) > 0:
        erbast_sizes = np.array([world.erbast_density(pos) * 10 for pos in erbast_positions])
        if erbast_positions.ndim == 1:
            erbast_scatter.set_offsets([erbast_positions[1], erbast_positions[0]])
        else:
            erbast_scatter.set_offsets(erbast_positions[:, [1, 0]])
        erbast_scatter.set_sizes(erbast_sizes)
    else:
        erbast_scatter.set_offsets(np.empty((0, 2)))
        erbast_scatter.set_sizes([])

    # Handling Carviz updates for unidimensional and multidimensional arrays
    if len(carviz_positions) > 0:
        carviz_sizes = np.array([world.carviz_density(pos) * 10 for pos in carviz_positions])
        if carviz_positions.ndim == 1:
            carviz_scatter.set_offsets([carviz_positions[1], carviz_positions[0]])
        else:
            carviz_scatter.set_offsets(carviz_positions[:, [1, 0]])
        carviz_scatter.set_sizes(carviz_sizes)
    else:
        carviz_scatter.set_offsets(np.empty((0, 2)))
        carviz_scatter.set_sizes([])

    return None


# Main function, runs the simulation
if __name__ == "__main__":
    world = World(size=NUMCELLS, ground_ratio=0.8)
    gui = PlanisussGUI(world)
    gui.run()