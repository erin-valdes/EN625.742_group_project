import pandas as pd
import numpy as np
from os import path

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
    df.drop(columns=badCols, inplace=True)
    return(df)

def convertToDateTime( row, col):
    '''
    Helper function for converting dates as done in existing code
    '''
    date_part, day_fraction = row[col].split('.')
    date_dt = pd.to_datetime(date_part, format='%Y-%m-%d')
    day_fraction_td = pd.to_timedelta(float('0.' + day_fraction) * 24, unit='hours')
    return date_dt + day_fraction_td

def processData( df ):
    '''
    Take in a df and preprocess it
    '''
    # Filter the Low Data Columns
    df = filterLowDataColumns(df)

    # # Convert Dates as Necessary
    df['epoch_cal'] = df.apply(convertToDateTime, args=('epoch_cal',), axis=1)
    df['tp_cal'] = df.apply(convertToDateTime, args=('tp_cal',), axis=1)
    # df['first_obs_year'] = pd.to_numeric(df['first_obs'].apply(lambda x: re.split('-|/', x)[0]), errors='coerce').astype('Int64')
    # df['first_obs_month'] = pd.to_numeric(df['first_obs'].apply(lambda x: re.split('-|/', x)[1]), errors='coerce').astype('Int64')
    # df['last_obs_year'] = pd.to_numeric(df['last_obs'].apply(lambda x: re.split('-|/', x)[0]), errors='coerce').astype('Int64')
    # df['last_obs_month'] = pd.to_numeric(df['last_obs'].apply(lambda x: re.split('-|/', x)[1]), errors='coerce').astype('Int64')
    # date_jd = ['epoch','tp']
    # df[date_jd] = df[date_jd].apply(pd.to_numeric)

    print(df.head())
    return(df)

def getAndProcessData( fname ):
    '''
    Take in a filename assumed to be a csv file
    load the data, process the data, return the 
    processed dataframe
    '''
    df = getData(fname)
    df = processData(df)
    return(df)