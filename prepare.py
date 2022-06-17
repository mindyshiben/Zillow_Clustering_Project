import pandas as pd
import numpy as np
from datetime import datetime
from env import get_db_url
import acquire 

def remove_outliers(df):

    '''
    This function takes a dataframe and applies several parameters to clean the data in useable form including renmaming
    columns, removing outliers, and changing data types. A cleaned dataframe is returned.
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
    df = df.dropna(thresh=df.shape[0]*0.2,how='all',axis=1)

    return df

def handle_nulls(df): 

    df = df.drop(columns=['calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'id', 'id.1'], axis=1)
    df = df.drop(columns=['buildingqualitytypeid', 'regionidcity', 'regionidzip', 'regionidneighborhood', 'roomcnt', 'unitcnt'], axis=1)
    df = df.drop(columns=['numberofstories','structuretaxvaluedollarcnt', 'taxamount', 'assessmentyear'],  axis=1)
    df = df.drop(columns=['airconditioningdesc', 'airconditioningtypeid', 'heatingorsystemdesc', 'heatingorsystemtypeid', 'regionidcounty'], axis=1)
    df = df.drop(columns=['propertyzoningdesc','censustractandblock', 'rawcensustractandblock'], axis=1)
    df[['garagecarcnt', 'garagetotalsqft']] = df[['garagecarcnt', 'garagetotalsqft']].fillna(0)
    df['poolcnt'] = df['poolcnt'].fillna(0)

    return df

def rename_columns(df):

    df = df.dropna()
    df.rename(columns={'bedroomcnt': 'bedrooms',
                   'lotsizesquarefeet': 'lot_size', 
                   'bathroomcnt': 'bathrooms', 
                   'calculatedfinishedsquarefeet': 'square_feet', 
                   'yearbuilt': 'year_built',
                    'garagecarcnt': 'garages',
                    'garagetotalsqft':'garage_size',
                    'poolcnt': 'has_pool',
                    'logerror': 'log_error',
                   'transactiondate': 'transaction_date',
                   'taxdollarvaluecount': 'tax_value'
                  }, inplace=True)
    df['year_built'] = df['year_built'].astype('int')
    df['fips'] = df['fips'].astype('int')
    df['square_feet'] = df['square_feet'].astype('int')
    df['county'] = df['fips'].replace({6037: 'los_angeles', 6059: 'orange', 6111: 'ventura'})
    df['lot_size'] = df.lot_size.astype(int)
    df['garages'] = df.garages.astype(int)
    df['garage_size'] = df.garage_size.astype(int)
    df['has_pool'] = df.has_pool.astype(bool)
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


    


    