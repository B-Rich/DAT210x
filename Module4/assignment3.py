import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper
from sklearn.decomposition import PCA
matplotlib.style.use('ggplot')

scaleFeatures = True

df = pd.read_csv('./Datasets/kidney_disease.csv', index_col=0)
df = df.dropna(axis=0)

# Create some color coded labels; the actual label feature
# will be removed prior to executing PCA, since it's unsupervised.
# You're only labeling by color so you can see the effects of PCA
labels = ['red' if i == 'ckd' else 'green' for i in df.classification]


# .. your code here ..
# df = df.drop(labels=['classification', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'], axis=1)
df.rc = pd.to_numeric(df.rc, errors='coerce')
df.wc = pd.to_numeric(df.wc, errors='coerce')
df.pcv = pd.to_numeric(df.pcv, errors='coerce')
df = pd.get_dummies(df, columns=['classification', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])

if scaleFeatures:
    df = helper.scaleFeatures(df)

pca = PCA(n_components=2)
pca.fit(df)
T = pca.transform(df)

# Plot the transformed data as a scatter plot. Recall that transforming
# the data will result in a NumPy NDArray. You can either use MatPlotLib
# to graph it directly, or you can convert it to DataFrame and have pandas
# do it for you.
#
# Since we've already demonstrated how to plot directly with MatPlotLib in
# Module4/assignment1.py, this time we'll convert to a Pandas Dataframe.
#
# Since we transformed via PCA, we no longer have column names. We know we
# are in P.C. space, so we'll just define the coordinates accordingly:
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()
