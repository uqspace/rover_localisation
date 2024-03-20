import matplotlib.pyplot as plt
import math

# Function to convert polar coordinates to Cartesian coordinates
def polar_to_cartesian(length, angle):
    x = length * math.sin(angle)
    y = length * math.cos(angle)
    return x, y

def generate_map():
    # Read points from the file
    points_file = "points2.txt"
    points = []
    with open(points_file, "r") as file:
        for line in file:
            length, angle = map(float, line.split())
            points.append((length, angle))

    # Initialize starting point at the bottom middle of the grid
    start_x = 0
    start_y = 0

    # Plot each line
    for length, angle in points:
        # Convert polar coordinates to Cartesian coordinates
        end_x, end_y = polar_to_cartesian(length, angle)

        # Adjust end coordinates relative to starting point
        end_x += start_x
        end_y += start_y

        # Plot line
        plt.plot([start_x, end_x], [start_y, end_y])

        # Update starting point for the next line
        start_x = end_x
        start_y = end_y

    # Set axis limits and labels
    plt.xlim(-4, 4)
    plt.ylim(0, 3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Theseus III Map')
    plt.grid(True)

    # Show plot
    plt.show()
