import matplotlib.pylab as plt
import numpy
import scipy.cluster.hierarchy as hcluster
import HULib
import pandas as pd
import numpy as np
import parameters
import hdbscan
import seaborn as sns
import time
plot_kwds = {'alpha' : 0.25, 's' : 2, 'linewidths':0}


def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)



myKeys=[
        ('DEC', 'galaxy_DR12v5_LOWZ_North.fits'),
        ('DEC', 'galaxy_DR12v5_CMASS_North.fits'),
        ('DEC', 'galaxy_DR12v5_LOWZ_South.fits'),
        ('DEC', 'galaxy_DR12v5_CMASS_South.fits'),
        ('RA', 'galaxy_DR12v5_LOWZ_North.fits'),
        ('RA', 'galaxy_DR12v5_CMASS_North.fits'),
        ('RA', 'galaxy_DR12v5_LOWZ_South.fits'),
        ('RA', 'galaxy_DR12v5_CMASS_South.fits')]


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
fig.subplots_adjust(hspace=.4)

key,value=myKeys[1]

myGalaxy = HULib.get_BOSS_data(parameters.sdssAddress + value)
myGalaxy = HULib.fix_BOSS_data(myGalaxy)
data=pd.DataFrame({'Me' : myGalaxy.groupby( ['alpha', key])['Me'].sum()}).reset_index()
data = data[[0, 2]]

data.plot(ax=axes[0], x='alpha', y='Me', legend=False,xlim=[0,0.7],ylim=[0,0.017],style='.')
axes[0].set_title(value + ' ' + key )
axes[0].set_ylabel("Intensity")
axes[0].set_xlabel("alpha")
hdbscan_ = hdbscan.HDBSCAN()
# ind=(data.alpha >= data.alpha.min()) & (data.alpha <= data.alpha.max())
ind=(data.alpha >= 0.25) & (data.alpha <= 0.38)
yy=data[ind]*1E6
hdbscan_data =  hdbscan_.fit(yy)
plot_clusters(yy.values, hdbscan.HDBSCAN, (), {'min_cluster_size':5})
frame = plt.gca()
fig = plt.gcf()
myMax=np.max(yy.Me)
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)
frame.axes.set_ylim([0,myMax])
plt.show()

yy['group']=hdbscan_data

a=1