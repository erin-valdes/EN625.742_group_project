import matplotlib.pyplot as plt


def ScatterPlot2d(x1, y1, x2, y2, lab1, lab2, title='', xlabel='', ylabel=''):
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
    plt.scatter(x2, y2, c='red', label=lab2, alpha=0.25)
    
    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    fname = f'scatter_{lab1}_{lab2}.png'
    plt.savefig( fname )