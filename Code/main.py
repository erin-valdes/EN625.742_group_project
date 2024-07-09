from os import getcwd
from os import path
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from pandas.plotting import scatter_matrix
from Charts import charts
from Utils import utils


def analyzeData(df):
    '''
    Do some statistical analysis on the data and make some charts
    '''
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

    PHA_Y = len(df[df['pha'] == 'Y'])
    PHA_N = len(df[df['pha'] == 'N']) 
    print(f'Potentially Harmful Y:{PHA_Y}    N:{PHA_N}')
    
    
    # # Plotting Histograms and Scatter Matrices
    # df.hist(figsize=(20, 15))
    # plt.tight_layout()  # Adjusts subplot params for a nice fit
    # plt.savefig('hist.png')
    # plt.clf()
    # scatter_matrix(df, figsize=(20, 15))
    # plt.savefig( 'scattermatrix.png' )
    # plt.clf()

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
    Initial Take at a 3d Cluster
    '''
    x1 = PHA_Y['t_jup'].to_numpy()
    y1 = PHA_Y['moid_jup'].to_numpy()
    z1 = PHA_Y['e'].to_numpy()
    x2 = PHA_N['t_jup'].to_numpy()
    y2 = PHA_N['moid_jup'].to_numpy()
    z2 = PHA_N['e'].to_numpy()
    charts.ScatterPlot3d(x1, y1, z1, x2, y2, z2, 'PHA: Y', 'PHA: N', '', 'Jupiter Tisserand Parameter', 'Jupiter Minimum Orbit Intersection Dist (au)', 'Eccentricity', 'scatter_t_jup_moid_jup_e.png')


    return()

if __name__ == '__main__':
    main()