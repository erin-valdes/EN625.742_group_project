from Utils import utils
from os import getcwd
from os import path



def main():
    '''
    Main Function for analysis
    '''
    fname = path.abspath(path.join(getcwd(), 'sbdb_query_results.csv'))
    df = utils.getAndProcessData(fname)
    return()

if __name__ == '__main__':
    main()