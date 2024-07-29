from os import getcwd
from os import path
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from pandas.plotting import scatter_matrix
from Charts import charts
from Utils import utils
from Models import models

# FUNCTIONS TO PLOT POTENTIAL RELATIONSHIPS
def Plotting( df ):
    '''
    Function to do all the plotting as part of the pre-modeling analysis
    '''
    PHA_Y = len(df[df['pha'] == 'Y'])
    PHA_N = len(df[df['pha'] == 'N']) 
    print(f'Potentially Harmful Y:{PHA_Y}    N:{PHA_N}')
    
    
    # # Plotting Histograms and Scatter Matrices
    # df.hist(figsize=(20, 15))
    # plt.tight_layout()  # Adjusts subplot params for a nice fit
    # plt.savefig('hist.png')
    # plt.clf()

    temp = df[ [c for c in df.columns if 'encoded' not in c] ]
    scatter_matrix(temp, figsize=(20, 15))
    plt.savefig( 'scattermatrix.png' )
    plt.clf()

    # Plotting Scatter Plots Color Coded by PHA Y or N
    PHA_Y = df[df['pha'] == 'Y']
    PHA_N = df[df['pha'] == 'N']


    '''
    This combo of eccentricity and absolute magnitude seem to exhibit promising results for
    a classification exercise.... logistic or QDA
    '''
    x1 = PHA_Y['e'].to_numpy()
    y1 = PHA_Y['H'].to_numpy()
    x2 = PHA_N['e'].to_numpy()
    y2 = PHA_N['H'].to_numpy()
    charts.ScatterPlot2d(x1, y1, x2, y2, 'PHA: Y', 'PHA: N', '', 'Eccentricity', 'Absolute Magnitude', 'scatter_e_H.png' )

    '''
    This combo of eccentricity and the MOID for Jupiter also seems to exhibit non-linear results
    for classification
    '''
    x1 = PHA_Y['e'].to_numpy()
    y1 = PHA_Y['moid_jup'].to_numpy()
    x2 = PHA_N['e'].to_numpy()
    y2 = PHA_N['moid_jup'].to_numpy()
    charts.ScatterPlot2d(x1, y1, x2, y2, 'PHA: Y', 'PHA: N', '', 'Eccentricity', 'Jupiter Minimum Orbit Intersection Dist (au)', 'scatter_e_moid_jup.png' )

    '''
    Theres an interesting almost sinusoidal pattern here and all the PHAs appear below the line
    '''
    x1 = PHA_Y['w'].to_numpy()
    y1 = PHA_Y['q'].to_numpy()
    x2 = PHA_N['w'].to_numpy()
    y2 = PHA_N['q'].to_numpy()
    charts.ScatterPlot2d(x1, y1, x2, y2, 'PHA: Y', 'PHA: N', '', 'Argument of Perihelion (deg)', 'Perihelion Distance (au)', 'scatter_w_q.png' )

    '''
    Theres an interesting almost sinusoidal pattern here and all the PHAs appear below the line
    '''
    x1 = PHA_Y['H'].to_numpy()
    y1 = PHA_Y['moid_jup'].to_numpy()
    x2 = PHA_N['H'].to_numpy()
    y2 = PHA_N['moid_jup'].to_numpy()
    charts.ScatterPlot2d(x1, y1, x2, y2, 'PHA: Y', 'PHA: N', '', 'Jupiter Minimum Orbit Intersection Dist (au)', 'Absolute Magnitude', 'scatter_H_moid_jup.png' )

    '''
    Initial Take at a 3d Cluster using Absolute Magnitude, Jupiter MOID, and Eccentricity
    '''
    x1 = PHA_Y['H'].to_numpy()
    y1 = PHA_Y['moid_jup'].to_numpy()
    z1 = PHA_Y['e'].to_numpy()
    x2 = PHA_N['H'].to_numpy()
    y2 = PHA_N['moid_jup'].to_numpy()
    z2 = PHA_N['e'].to_numpy()
    charts.ScatterPlot3d(x1, y1, z1, x2, y2, z2, 'PHA: Y', 'PHA: N', '', 'Absolute Magnitude', 'Jupiter Minimum Orbit Intersection Dist (au)', 'Eccentricity', 'scatter_H_moid_jup_e.png')


    x = PHA_Y['om'].to_numpy()
    y = PHA_Y['w'].to_numpy()
    z = PHA_Y['e'].to_numpy()
    charts.ClustedPlot3d(x, y, z, '', 'Longitude of the Ascending Node (deg)', 'Argument of Perihelion (deg)', 'Eccentricity', 'cluster_om_w_e.png')


# FUNCTIONS FOR MODELLING RELATIONSHIPS TO CLASSIFY PHA
def AnalysisV1( df ):
    '''
        Relationship: Eccentricity (e) vs. Absolute Magnitude (H)
        -   LDA: Use a linear decision boundary where we assume equal covariances to test if we can separate the data
        -   Logistic Regression: Use a linear decision boundary with lighter assumptions than LDA to check if we can separate the data better
        -   SVM (with Kernels): Test an SVM with varying Kernels.  Maybe in higher dimensions we have more easily separable data
        
        X is a 2xN matrix where the inputs are e and H
        y is the response variable which is PHA y/n
    '''    
    X = df[ ['e', 'H'] ]
    y = df[ ['pha_encoded'] ]
    print(f'{X.head()}\n{y.head()}')
    [avg, std, Logistic] = models.Logistic(X, y)
    
    return()


def main():
    '''
    Main Function for analysis
    '''
    fname = path.abspath(path.join(getcwd(), 'sbdb_query_results.csv'))
    full_df = utils.getAndPreProcessData(fname)
    full_df = utils.scaleColumns( full_df, [
        'H','n','moid_jup','t_jup','rms','q','e','w','om'       
    ] )

    # Subset the data even more, we dont really care for the sigma cols and for analysis dont need the un_encoded variables either
    # Take a sample of 10% of the data which we will use for the exploratory part of our analysis
    columnsForAnalysis = [ 
        'H','n','moid_jup','t_jup','rms','q','e','w','om', 'pha', 'neo',
        'neo_encoded', 'pdes_encoded', 'pha_encoded', 'class_encoded' 
    ]
    df = full_df[columnsForAnalysis]
    df = df.sample(frac=0.1)
    df.dropna( inplace=True)

    print(df.head())

    # Create and Save Pre-analysis plots
    # Plotting(df)


    '''
    Here I detail what boudnaries and algorithms I actually want to test and how I'd like to test them.  For every model we implement, 
    assume 10-fold cross validation is used.  

    Relationship: Eccentricity (e) vs. Absolute Magnitude (H)
        -   LDA: Use a linear decision boundary where we assume equal covariances to test if we can separate the data
        -   Logistic Regression: Use a linear decision boundary with lighter assumptions than LDA to check if we can separate the data better
        -   SVM (with Kernels): Test an SVM with varying Kernels.  Maybe in higher dimensions we have more easily separable data

    Relationship: Eccentricity (v) vs. Jupiter MOID (moid_jup_)
        -   QDA: See if a quadratic decision boundary can separate data better than a linear
        -   SVM (with Kernels):  See if higher dimensional spaces can separate better than a QDA
    
    Relationship: Arg of Perihelion (w) vs. Perihelion Distance (q)
        -   SVM with Gaussian Kernel
        -   SVM with a periodic kernel 
    '''
    AnalysisV1(df)

    return()

if __name__ == '__main__':
    main()