import pandas as pd
import numpy as np
from datetime import datetime
from env import get_db_url
import acquire 
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split

def remove_outliers(df):

    '''
    This function takes a dataframe and applies several parameters to clean the data in useable form, specifically removing outliers.
    '''

    df = df[df.bathroomcnt >= 1]
    df = df[df.bathroomcnt <= 5]
    df = df[df.bedroomcnt >= 1]
    df = df[df.bedroomcnt <= 5]
    df = df[df.logerror < 0.5]
    df = df[df.logerror > (-0.31)]
    df = df[df.yearbuilt >= 1910]
    df = df[df.calculatedfinishedsquarefeet >= 650]
    df = df[df.calculatedfinishedsquarefeet <= 5500]
    df = df[df.taxvaluedollarcnt > 40000.0]
    df = df[df.taxvaluedollarcnt < 3000000.0]
    df = df[df.lotsizesquarefeet< 50000]
    df = df[(df.propertycountylandusecode == '0100') | (df.propertycountylandusecode == '122') | (df.propertycountylandusecode == '0101') | (df.propertycountylandusecode == '1111') | (df.propertycountylandusecode == '1') | (df.propertycountylandusecode == '1110') | (df.propertycountylandusecode == '0104')]
    df = df.dropna(thresh=df.shape[0]*0.2,how='all',axis=1)

    return df

def handle_nulls(df): 

    '''
    This function takes a dataframe and applies several parameters to clean the data in useable form, specifically handling null values.
    '''

    df = df.drop(columns=['calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'id'])
    df = df.drop(columns=['buildingqualitytypeid', 'regionidcity', 'regionidzip', 'regionidneighborhood', 'roomcnt', 'unitcnt'])
    df = df.drop(columns=['garagecarcnt', 'garagetotalsqft', 'poolcnt', 'numberofstories','structuretaxvaluedollarcnt', 'taxamount', 'assessmentyear'])
    df = df.drop(columns=['airconditioningdesc', 'airconditioningtypeid', 'heatingorsystemdesc', 'heatingorsystemtypeid', 'regionidcounty'])
    df = df.drop(columns=['propertyzoningdesc','censustractandblock', 'rawcensustractandblock'])

    return df

def rename_columns(df):
    
    '''
    This function takes a dataframe and applies several parameters to clean the data in useable form, specifically renaming columns.
    '''

    df = df.dropna()
    df.rename(columns={'bedroomcnt': 'bedrooms',
                   'lotsizesquarefeet': 'lot_size', 
                   'bathroomcnt': 'bathrooms', 
                   'calculatedfinishedsquarefeet': 'square_feet', 
                   'yearbuilt': 'year_built',
                    'logerror': 'log_error',
                   'transactiondate': 'transaction_date',
                   'taxdollarvaluecount': 'tax_value'
                  }, inplace=True)
    df['year_built'] = df['year_built'].astype('int')
    df['fips'] = df['fips'].astype('int')
    df['square_feet'] = df['square_feet'].astype('int')
    df['county'] = df['fips'].replace({6037: 'los_angeles', 6059: 'orange', 6111: 'ventura'})
    df['lot_size'] = df.lot_size.astype(int)
    df['transaction_date'] = pd.to_datetime(df.transaction_date)
    
    return df

def prepare_locs(df):

    '''
    This function takes a dataframe and manipulates the format of latitude and longitude columns 
    to be in the correct form. The dataframe is returned with said changes.
    '''
    
    long = pd.DataFrame(df['longitude'])
    for c in long:
        long[c] = (long[c] / 1000000)
    lat = pd.DataFrame(df['latitude'])
    for c in lat:
        lat[c] = (lat[c] / 1000000)
    df.drop(columns = ['latitude', 'longitude'], inplace=True)
    df = pd.concat([df, lat, long], axis=1)

    return df


def split_zillow_data(df):
    
    '''
    This function takes in a dataframe and splits it into three subgroups: train, test, validate
    for proper evalution, statistical testing, and modeling. Three dataframes are returned.
    '''

    train, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    return train, validate, test

def make_zillow_bins(train, validate, test, train_pos, train_neg):

    ''' this function takes in dataframes and creates categorial bins on selected features'''

    bins = [-.6, -0.04627, 0, 0.057585, .6]
    labels = ['high_neg','low_neg','low_pos','high_pos']

    train['log_error_bin'] = pd.cut(train['log_error'], bins=bins, labels=labels)
    validate['log_error_bin'] = pd.cut(validate['log_error'], bins=bins, labels=labels)
    test['log_error_bin'] = pd.cut(test['log_error'], bins=bins, labels=labels)
    train_pos['log_error_bin'] = pd.cut(train_pos['log_error'], bins=bins, labels=labels)
    train_neg['log_error_bin'] = pd.cut(train_neg['log_error'], bins=bins, labels=labels)


    bins = [1900, 1930, 1950, 1970, 2000, 2020]
    labels = labels = ['1910-30' ,'1930-50', '1950-70', '1970-2000', '2000-20']
    train['year_bin'] = pd.cut(train['year_built'], bins=bins, labels=labels)
    validate['year_bin'] = pd.cut(train['year_built'], bins=bins, labels=labels)
    test['year_bin'] = pd.cut(train['year_built'], bins=bins, labels=labels)
    train_pos['year_bin'] = pd.cut(train_pos['year_built'], bins=bins, labels=labels)
    train_neg['year_bin'] = pd.cut(train_neg['year_built'], bins=bins, labels=labels)

    bins = [0, 1214, 1497, 1850, 2430, 5600]
    labels = ['XS', 'S', "M", 'L', "XL"]
    train['square_feet_bin'] = pd.cut(train['square_feet'], bins=bins, labels=labels)
    validate['square_feet_bin'] = pd.cut(validate['square_feet'], bins=bins, labels=labels)
    test['square_feet_bin'] = pd.cut(test['square_feet'], bins=bins, labels=labels)
    train_pos['square_feet_bin'] = pd.cut(train_pos['square_feet'], bins=bins, labels=labels)
    train_neg['square_feet_bin'] = pd.cut(train_neg['square_feet'], bins=bins, labels=labels)

    bins = [0, 5272, 6299, 7368, 9518, 100000000]
    labels = ['XS', 'S', "M", 'L', "XL"]
    train['lot_size_bin'] = pd.cut(train['lot_size'], bins=bins, labels=labels)
    validate['lot_size_bin'] = pd.cut(validate['lot_size'], bins=bins, labels=labels)
    test['lot_size_bin'] = pd.cut(test['lot_size'], bins=bins, labels=labels)
    train_pos['lot_size_bin'] = pd.cut(train_pos['lot_size'], bins=bins, labels=labels)
    train_pos['lot_size_bin'] = pd.cut(train_pos['lot_size'], bins=bins, labels=labels)


    bins = [0, 54240, 159758, 279118, 448013, 3817215]
    labels = ['very_low','low', 'medium ', 'high', 'very_high']
    train['lot_value_bin'] = pd.cut(train['landtaxvaluedollarcnt'], bins=bins, labels=labels)
    validate['lot_value_bin'] = pd.cut(validate['landtaxvaluedollarcnt'], bins=bins, labels=labels)
    test['lot_value_bin'] = pd.cut(test['landtaxvaluedollarcnt'], bins=bins, labels=labels)
    train_pos['lot_value_bin'] = pd.cut(train_pos['landtaxvaluedollarcnt'], bins=bins, labels=labels)
    train_neg['lot_value_bin'] = pd.cut(train_neg['landtaxvaluedollarcnt'], bins=bins, labels=labels)
    

    bins = [500, 160726, 299498, 445086, 667000, 3000000]
    labels = ['very_low','low', 'medium ', 'high', 'very_high']
    train['value_bin'] = pd.cut(train['taxvaluedollarcnt'], bins=bins, labels=labels)
    validate['value_bin'] = pd.cut(train['taxvaluedollarcnt'], bins=bins, labels=labels)
    test['value_bin'] = pd.cut(train['taxvaluedollarcnt'], bins=bins, labels=labels)
    train_pos['value_bin'] = pd.cut(train_pos['taxvaluedollarcnt'], bins=bins, labels=labels)
    train_neg['value_bin'] = pd.cut(train_neg['taxvaluedollarcnt'], bins=bins, labels=labels)