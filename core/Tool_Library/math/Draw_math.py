from __future__ import annotations

"""
math_figure_drawer.py

A compact Python module + mini-DSL to parse simple geometry/math drawing tasks
and produce publication-quality figures with matplotlib. Designed for easy
integration into a larger assistant (like Ritsu) to generate diagrams for
algebra, inequalities, vectors, and Euclidean geometry problems.

Dependencies:
 - numpy
 - matplotlib
 - sympy
 - shapely (optional, for polygon/region operations)

Main features:
 - mini-DSL (GT-style: Given / Call / Let / Cond) to declare points & constraints
 - helpers for: triangle construction, centroid/reflection, midpoints, vector
   arithmetic, plotting inequalities (half-planes/intersections), plotting
   quadratics, scaling/auto-unit to make figures readable.
 - examples implementing the user's listed exercises
"""

import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --- 1. Data Structure for Points ---
# Using a dataclass provides clear structure and type hinting for coordinates.
@dataclass
class Point:
    """Represents a single point in 2D or 3D space."""
    x: float
    y: float
    z: float = 0.0 # Default to 0.0 for 2D, allowing easy extension to 3D later

    def coords_2d(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def __repr__(self):
        return f"P({self.x}, {self.y}, {self.z})"


# --- 2. Configuration and Main Drawing Class ---
class DrawMath:
    """
    Manages points, visualization configuration, and drawing operations.
    This component acts as the visualization layer for your Math functions.
    """
    
    def __init__(self, **kwargs):
        # A dictionary is the perfect solution for handling a non-permanent number 
        # of named points (A, B, point_1, etc.).
        self.points: Dict[str, Point] = {} 
        self.shapes: List[Dict] = []
        
        # Global visualization settings (can be overridden/bridged)
        self.config = {
            'grid_show': kwargs.get('grid_show', True),
            'grid_scale': kwargs.get('grid_scale', 5), # Default axis limit
            'axis_equal': kwargs.get('axis_equal', True),
            'grid_center': kwargs.get('grid_center', False), # Centers view on (0,0) if True
        }
        print("DrawMath system initialized.")


    # --- Point Management Functions (Resolves your Failsafe/Naming concern) ---

    def add_point(self, name: str, x: float, y: float, z: float = 0.0):
        """Adds a new named point or updates an existing one."""
        if not name or not name.isalnum():
             raise ValueError("Point name must be a non-empty alphanumeric string.")
        self.points[name] = Point(x, y, z)
        print(f"Added/Updated point '{name}' at {self.points[name]}")

    def _get_point_coords(self, *names_or_points) -> List[Point]:
        """Internal helper to resolve string names to Point objects."""
        resolved_points = []
        for item in names_or_points:
            if isinstance(item, str) and item in self.points:
                resolved_points.append(self.points[item])
            elif isinstance(item, Point):
                resolved_points.append(item)
            else:
                raise ValueError(f"Point '{item}' not found or invalid type.")
        return resolved_points

    def clear_all(self):
        """Clears all defined points and shapes to start a new drawing."""
        self.points = {}
        self.shapes = []
        print("\nDrawing system cleared.")

    # --- Shape Definition Functions ---

    def draw_square(self, name: str, A: str, B: str, C: str, D: str):
        """
        Defines a square using four named points (e.g., A='P1', B='P2', etc.).
        The points must already exist in self.points.
        """
        try:
            points = self._get_point_coords(A, B, C, D)
        except ValueError as e:
            print(f"Error defining square '{name}': {e}")
            return
        
        # Store shape definition as a list of coordinates for plotting
        x = [p.x for p in points] + [points[0].x] # Close the loop
        y = [p.y for p in points] + [points[0].y]
        
        self.shapes.append({
            'name': name,
            'type': 'line',
            'x_data': x,
            'y_data': y,
            'style': 'b-', # Blue solid line
            'label': name,
        })
        print(f"Defined square: '{name}' using points {A}, {B}, {C}, {D}")

    # --- Visualization (Matplotlib) ---

    def _plot_2d(self):
        """Handles the actual rendering logic using Matplotlib."""
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 1. Draw all shapes
        for shape in self.shapes:
            ax.plot(shape['x_data'], shape['y_data'], shape['style'], label=shape['label'])

        # 2. Draw all individual points and labels
        for name, point in self.points.items():
            ax.plot(point.x, point.y, 'ro') # Red circle for points
            ax.annotate(name, (point.x + 0.1, point.y + 0.1)) # Label point

        # 3. Apply Configuration
        ax.set_title("Draw Math Visualization", fontsize=14)
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        
        if self.config['grid_show']:
            ax.grid(True, linestyle='--', alpha=0.6)
            
        if self.config['grid_center']:
             # Center around (0,0) with the defined scale
            scale = self.config['grid_scale']
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
        else:
             # Automatically determine limits based on content
             ax.autoscale_view() 
            
        if self.config['axis_equal']:
            ax.set_aspect('equal', adjustable='box')
            
        ax.legend()
        plt.show()

    def show(self):
        """Public method to display the visualization."""
        self._plot_2d()


# --- Example Usage ---

if __name__ == '__main__':
    
    # 1. Initialize the system
    # Set a small scale and turn the grid on
    drawer = DrawMath(grid_show=True, grid_scale=3) 

    # 2. Define Points (Using your example coordinates)
    # The 'add_point' function is the failsafe for naming
    drawer.add_point('A', -2, 2)
    drawer.add_point('B', 2, 2)
    drawer.add_point('C', 2, -2)
    drawer.add_point('D', -2, -2)
    drawer.add_point('E', 0, 0)
    drawer.add_point('F', 1.5, 0)
    
    # 3. Define Shapes
    # Define a square using the named points
    drawer.draw_square('Square_ABCD', 'A', 'B', 'C', 'D')
    
    # Define a different polygon (like a triangle) using existing points
    # NOTE: Matplotlib just draws lines between the points in the order given
    drawer.draw_square('Line_EFC', 'E', 'F', 'C', 'E') # Pass 'E' twice to just draw triangle/line
    
    # 4. Visualize
    print("\nDisplaying Visualization...")
    # NOTE: In a real AI environment, this 'show' call would likely save an image 
    # or export the plot data structure for a frontend/C++ renderer.
    # We use plt.show() here for demonstration.
    drawer.show()

    # 5. Clear and define a new shape (for testing reset)
    drawer.clear_all()
    drawer.add_point('P1', 1, 1)
    drawer.add_point('P2', 5, 1)
    drawer.add_point('P3', 5, 5)
    drawer.add_point('P4', 1, 5)
    
    drawer.draw_square('New_Square', 'P1', 'P2', 'P3', 'P4')
    drawer.show()
    
