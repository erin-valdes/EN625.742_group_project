import pandas as pd
import numpy as np
from os import path
import re
from sklearn.preprocessing import scale, LabelEncoder
import category_encoders as ce

def getData(fname=''):
    '''
    Take in a filename (if not already specified), load the 
    data into a pandas df and return the df
    '''
    if path.isfile(fname):
        return( pd.read_csv(fname) )
    else:
        print( 'ERROR: Bad file, check your filename' )
        return( None )

def filterLowDataColumns(df, cutoff=0.2):
    '''
    Takes in a cutoff expressed as a percent
    sum the number of null rows in a column.
    if it is greater than cutoff% of the total
    number of rows, boot the column.
    '''
    badCols = df.columns[df.isna().sum() > len(df) * cutoff ]
    # print(badCols)
    df.drop(columns=badCols, inplace=True)
    return(df)

def EncodeCategoricalVariables( df, target_col, cat_columns=[]):
    '''
    Take in an array of categorical variables to encode and
    return the df with the columns encoded

    FIXME:  We need to discern the encodings and return a dictionary of the encodings to map them back
    '''
    for c in cat_columns:
        # LE = LabelEncoder()
        # use catboost encoder
        CE = ce.CatBoostEncoder()
        df[ c + '_encoded' ] = CE.fit_transform( df[c], df[target_col] )
    return( df )

def preProcessData( df ):
    '''
    Take in a df and preprocess it
    '''
    # Filter the Low Data Columns
    df = filterLowDataColumns(df)

    # Drop Unecessary Columns: Likely we wont need these for training our algos
    df.drop(
        columns=[
            'epoch_mjd', 'epoch', 'epoch_cal', 'first_obs', 'last_obs',  'producer'
        ],
        inplace=True
    )

    # Encode Categorical Variables
    cols = [ 'neo', 'pdes', 'pha', 'class', ]
    df = EncodeCategoricalVariables(df, cols)
    return(df)

def scaleColumns(df, columns=[], with_mean=True, with_std=True):
    '''
    Takes in a df and target columns (which should be numeric)
    and scales them.  The mean flag will center everything about
    the mean, the std flag will put it all in z-scores
    '''
    for c in columns:
        df[c] = scale(df[c], with_mean=with_mean, with_std=with_std)

    return(df)

def getAndPreProcessData( fname ):
    '''
    Take in a filename assumed to be a csv file
    load the data, process the data, return the 
    processed dataframe
    '''
    df = getData(fname)
    df = preProcessData(df)
    return(df)