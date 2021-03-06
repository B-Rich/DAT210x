import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import datetime

matplotlib.style.use('ggplot')  # Look Pretty

#
# TODO: To procure the dataset, follow these steps:
# 1. Navigate to: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
# 2. In the 'Primary Type' column, click on the 'Menu' button next to the info button,
#    and select 'Filter This Column'. It might take a second for the filter option to
#    show up, since it has to load the entire list first.
# 3. Scroll down to 'GAMBLING'
# 4. Click the light blue 'Export' button next to the 'Filter' button, and select 'Download As CSV'


def doKMeans(df):
    #
    # INFO: Plot your data with a '.' marker, with 0.3 alpha at the Longitude,
    # and Latitude locations in your dataset. Longitude = x, Latitude = y
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)
    latlon = df[['Longitude', 'Latitude']]
    kmeans = KMeans(n_clusters=7)
    kmeans_model = kmeans.fit(latlon)
    centroids = kmeans_model.cluster_centers_
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
    print centroids

#
# TODO: Load your dataset after importing Pandas
#
# .. your code here ..


#
# TODO: Drop any ROWs with nans in them
#
# .. your code here ..


#
# TODO: Print out the dtypes of your dset
#
# .. your code here ..


#
# Coerce the 'Date' feature (which is currently a string object) into real date,
# and confirm by re-printing the dtypes. NOTE: This is a slow process...
#
# .. your code here ..


df = pd.read_csv('./Datasets/Crimes_-_2001_to_present.csv')
df = df.dropna(axis=0)
df['Date'] = pd.to_datetime(df.Date, errors='coerce')
# INFO: Print & Plot your data
doKMeans(df)
doKMeans(df)
doKMeans(df)


#
# TODO: Filter out the data so that it only contains samples that have
# a Date > '2011-01-01', using indexing. Then, in a new figure, plot the
# crime incidents, as well as a new K-Means run's centroids.
#
# .. your code here ..
dt = datetime.datetime(2011, 1, 1)
print(df.dtypes)
df = df[df.Date > dt]
print(df.head())
doKMeans(df)
doKMeans(df)
doKMeans(df)


# INFO: Print & Plot your data
plt.show()
