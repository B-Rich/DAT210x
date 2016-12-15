import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from sklearn import manifold

import os

# Look pretty...
matplotlib.style.use('ggplot')


samples = []
image_dir = './Datasets/ALOI/32/'
for img_file in os.listdir(image_dir):
    img = misc.imread('{}{}'.format(image_dir, img_file))
    img = img.reshape(-1)
    samples.append(img)

df = pd.DataFrame.from_records(samples)

iso = manifold.Isomap(n_neighbors=8, n_components=3)
iso.fit(df)
T = iso.transform(df)

#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here ..


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here ..


#
# TODO: Convert the list to a dataframe
#
# .. your code here ..



#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here ..



#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here ..




#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here ..



# plt.show()
