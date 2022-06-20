import pandas as pd
import numpy as np
from env import get_db_url
import os

def get_zillow_data():

    '''
    This function acquires zillow data by accessing a SQL database and performing a SQL query to acquire
    selected zillow tables and columns and return it to a dataframe.
    '''

    filename = 'zillow.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        sql = """
        SELECT *
        FROM properties_2017
        JOIN predictions_2017 USING (parcelid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
        WHERE transactiondate LIKE '2017%%'
		and propertylandusetypeid = 261;
        """

        df = pd.read_sql(sql, get_db_url('zillow'))

        return df 