# Planisuss: A Simulation of Ecosystem Dynamics
-----
### Project Overview
Planisuss is an interactive ecological simulation inspired by Wa-Tor and Conway’s Game of Life. The simulation models the interactions between three primary entities in a simplified world: Vegetob (plants), Erbast (herbivores), and Carviz (carnivores). These species engage in various ecological processes such as population growth, resource competition, predator-prey relationships, and group behaviors.

The purpose of this project is to explore the dynamic relationships within this artificial ecosystem and provide real-time visual feedback through an intuitive graphical user interface (GUI).

### Key Entities in Planisuss
- **Vegetob:** Primary producers representing vegetation that grows and spreads across the landscape.
- **Erbast:** Herbivores that feed on Vegetob and move in herds.
- **Carviz:** Carnivores that hunt Erbast and move in prides.

### Features and Design Choices
- **World Initialization:** The world is represented as a grid of cells, initialized with water and ground cells. The landscape resembles a natural island, with Vegetob, Erbast, and Carviz randomly distributed across the map.
- **Species Interaction:** Each species has its own behavior, movement, and decision-making logic. Erbast graze on Vegetob, while Carviz hunt Erbast. Social behaviors, group actions, and decision-making are all incorporated into their movements and actions.
- **Simulation Dynamics:** The simulation progresses day by day, allowing users to observe the natural cycles of life, such as birth, death, and interactions between species. The world is updated in real-time, showing changes to the ecosystem.
- **Helper Functions:** Key functions for movement, evaluation of cell desirability, density calculations, and ensuring valid movements are implemented to drive the simulation.

### GUI and User Interaction
The project features an interactive Graphical User Interface (GUI) built with Tkinter, allowing users to:

- **Pause/Resume the Simulation:** Manage the flow of time within the simulation.
- **Zoom on Cells:** Inspect details about specific cells, including vegetation intensity and the number of creatures present.
- **Display Graphs:** Analyze trends in population, energy, and vegetation growth over time.
- **Adjust Simulation Parameters:** Modify key parameters like maximum herd/pride size, aging, energy thresholds, and more, to explore different ecological scenarios.
  
The GUI ensures that users are actively engaged with the simulation, providing a visual and analytical platform to observe and control the dynamics of the ecosystem.

### Key Design Benefits
- **Object-Oriented Design:** The project uses an object-oriented approach, making it modular and easy to maintain. Superclasses and inheritance are used to promote code reuse and simplify the simulation logic.
- **Real-Time Updates:** Animations and real-time updates provide a dynamic view of how the ecosystem evolves, offering immediate feedback for any changes made by the user.
- **Customizable Parameters:** Users can experiment with different simulation settings to explore how changes affect the dynamics of Planisuss.

### Limitations and Future Improvements
- **Resource Limitations:** The current simulation is 2D and lacks vertical movement and more complex terrain features. Future improvements could involve 3D structures to create a more immersive experience.
- **AI and Machine Learning:** A potential future direction includes incorporating machine learning techniques, such as genetic algorithms, to simulate evolution and adaptive behavior within the species.

### References
This simulation draws on Python libraries including **NumPy** (for numerical operations) and **Matplotlib** (for visualization). The GUI is built using **Tkinter** for cross-platform interaction. To run the project, ensure you have both the Python script and the image files in the same environment. Further documentation and resources:

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [Tkinter Documentation](https://docs.python.org/3/library/tk.html)
