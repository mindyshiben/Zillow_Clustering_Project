import pandas as pd
import numpy as np
from datetime import datetime
from env import get_db_url
import prepare
from IPython.display import Markdown, display
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import sklearn.preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans2, whiten


def create_yearsize_feature(train, train_pos, train_neg, validate, test):

    '''this function takes in dataframes used, scales square_feet and year_built function, and creates a new column on each dataframe with KMEANS clustering '''

    trainKMEANS = train[['square_feet', 'year_built']]
    trainposKMEANS = train_pos[['square_feet', 'year_built']]
    trainnegKMEANS = train_neg[['square_feet', 'year_built']]
    validateKMEANS = validate[['square_feet', 'year_built']]
    testKMEANS = test[['square_feet', 'year_built']]

    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(trainKMEANS)
    train_s = (trainKMEANS)
    train_ps = (trainposKMEANS)
    train_ns = (trainnegKMEANS)
    validate_s = (validateKMEANS)
    test_s = (testKMEANS)

    X = scaler.transform(trainKMEANS)
    Xpos = scaler.transform(trainposKMEANS)
    Xneg = scaler.transform(trainnegKMEANS)
    X1 = scaler.transform(validateKMEANS)
    X2 = scaler.transform(testKMEANS)

    X = pd.DataFrame(X, index=train_s.index, columns=train_s.columns)
    Xpos = pd.DataFrame(Xpos, index=train_ps.index, columns=train_ps.columns)
    Xneg = pd.DataFrame(Xneg, index=train_ns.index, columns=train_ns.columns)
    X1 = pd.DataFrame(X1, index=validate_s.index, columns=validate_s.columns)
    X2 = pd.DataFrame(X2, index=test_s.index, columns=test_s.columns)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)

    train['year_size_cluster'] = kmeans.predict(X)
    train_pos['year_size_cluster'] = kmeans.predict(Xpos)
    train_neg['year_size_cluster'] = kmeans.predict(Xneg)
    validate['year_size_cluster'] = kmeans.predict(X1)
    test['year_size_cluster'] = kmeans.predict(X2)

def create_loc_feature(train, validate, test):

    ''' this function takes in dataframes used,  and creates a new column on each dataframe with DBSCAN clustering using location data and the year_size_cluster column'''

    X = train[['latitude', 'longitude', 'year_size_cluster']]
    X1 = validate[['latitude', 'longitude', 'year_size_cluster']]
    X2 = test[['latitude', 'longitude', 'year_size_cluster']]

    clustering = DBSCAN(eps = .21, min_samples = 400).fit(X)
    clustering1 = DBSCAN(eps = .21, min_samples = 300).fit(X1)
    clustering2 = DBSCAN(eps = .21, min_samples = 200).fit(X2)

    train['DB_locs']=clustering.labels_
    validate['DB_locs']=clustering1.labels_
    test['DB_locs']=clustering2.labels_