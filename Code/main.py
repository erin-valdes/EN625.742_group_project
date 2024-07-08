from Utils import utils
from os import getcwd
from os import path
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from pandas.plotting import scatter_matrix


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
    df = utils.getAndPreProcessData(fname)
    df = utils.scaleColumns( df, [
        'H','n','moid_jup','t_jup','rms','q','e','w','om'       
    ] )
    



    df.hist(figsize=(20, 15))
    plt.tight_layout()  # Adjusts subplot params for a nice fit
    plt.savefig('hist.png')
    plt.clf()
    # scatter_matrix(df)
    plt.show()
    plt.savefig( 'scattermatrix.png' )
    plt.clf()
    return()

if __name__ == '__main__':
    main()