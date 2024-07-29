import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def ScatterPlot2d(x1, y1, x2, y2, lab1, lab2, title='', xlabel='', ylabel='', fname=''):
    """
    Plots two sets of data on the same scatter plot with different colors.
    
    Parameters:
    x1, y1: from the first dataset
    x2, y2: from the second dataset
    lab1: label for the first dataset
    lab2: label for the second dataset
    title: title of the plot (default is "Scatter Plot")
    xlabel: label for the x-axis (default is "X-axis")
    ylabel: label for the y-axis (default is "Y-axis")
    """
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x1, y1, c='blue', label=lab1, alpha=0.25)
    plt.scatter(x2, y2, c='red', label=lab2, alpha=0.1)
    
    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add a legend
    plt.legend()
    
    # Save the plot
    plt.savefig( fname )

def ScatterPlot3d(x1, y1, z1, x2, y2, z2, labels1, labels2, title='', xlabel='', ylabel='', zlabel='', fname=''):
    """
    Plots three sets of data on the same 3D scatter plot with different colors and 50% transparency.
    
    Parameters:
    x1, y1, z1: numpy arrays of x, y, z values for the first dataset
    x2, y2, z2: numpy arrays of x, y, z values for the second dataset
    x3, y3, z3: numpy arrays of x, y, z values for the third dataset
    labels1: label for the first dataset
    labels2: label for the second dataset
    labels3: label for the third dataset
    title: title of the plot (default is "3D Scatter Plot")
    xlabel: label for the x-axis (default is "X-axis")
    ylabel: label for the y-axis (default is "Y-axis")
    zlabel: label for the z-axis (default is "Z-axis")
    """
    # Create a 3D scatter plot with 50% transparency
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x1, y1, z1, c='blue', label=labels1, alpha=0.25)
    ax.scatter(x2, y2, z2, c='red', label=labels2, alpha=0.1)
    
    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    # Add a legend
    ax.legend()

    # Save the plot
    plt.savefig( fname )

def ClustedPlot3d(x, y, z, title='', xlab='', ylab='', zlab='', fname=''):
    '''
    Plots a clusted of 3d data
    
    Parameters:
    x, y, z: numpy arrays of x, y, z values for the dataset
    title: title of the plot (default is "3D Scatter Plot")
    xlabel: label for the x-axis (default is "X-axis")
    ylabel: label for the y-axis (default is "Y-axis")
    zlabel: label for the z-axis (default is "Z-axis")
    '''
    # Create a 3D scatter plot with 50% transparency
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x, y, z, c='red', alpha=0.25)
    
    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    
    # Save the plot
    plt.savefig( fname )
 