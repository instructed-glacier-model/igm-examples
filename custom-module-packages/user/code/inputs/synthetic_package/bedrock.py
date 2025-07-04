import numpy as np

def create_bedrock(X, Y):
    """
    Function to calculate the bedrock topography based on the given X and Y coordinates.
    
    Parameters:
    X (numpy.ndarray): 2D array of X coordinates.
    Y (numpy.ndarray): 2D array of Y coordinates.
    
    Returns:
    numpy.ndarray: 2D array representing the bedrock topography.
    """

    return 1000 + 0.15 * Y + ((X - 5000) ** 2) / 50000

def get_coordinates(length_x, length_y, resolution):
    """
    Function to generate the X and Y coordinates for a grid based on specified lengths and resolution.
    Units are assumed to match and be up to the discretion of the user.
    
    Parameters:
    length_x (int): The length of the X dimension.
    length_y (int): The length of the Y dimension.
    resolution (int): The resolution of the grid, default is 100.
    
    Returns:
    tuple: Two 2D arrays representing the X and Y coordinates of the grid.
    """
    
    
    x = np.arange(0, length_x) * resolution  # make x-axis
    y = np.arange(0, length_y) * resolution  # make y-axis

    X, Y = np.meshgrid(x, y)

    return x, y, X, Y