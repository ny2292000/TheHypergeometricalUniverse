import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster
import HULib

# generate 3 clusters of each around 100 points and an orphan vector
N=100
data = numpy.random.randn(3*N,2)
data[:N] += 5
data[-N:] += 10
data[-1:] -= 20

# clustering
thresh = 1.5
clusters = hcluster.fclusterdata(data, thresh, criterion="distance")

# plotting
plt.scatter(*numpy.transpose(data), c=clusters)
plt.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
plt.title(title)
plt.show()