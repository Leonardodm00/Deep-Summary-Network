# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 11:42:29 2025

@author: leona
"""

'''
This script implements the function 'Embedding accuracy' meant to provide scores on the
goodness of the deep NN in embedding temporal traces. The traces embedded will be of known condition (e.g. patho, control etc...)

How it works:
    1. **Generate Embeddings**: Pass your temporal traces through the network to get the n-dimensional embeddings for each trace.
    
    2. **Cluster the Embeddings**: Apply a clustering algorithm (e.g., K-Means, DBSCAN) to the generated embeddings. 
        The choice of algorithm can depend on the desired number of clusters or the density of your data.
    
    3. **Calculate the Scores**: Use the clustering results (and ground truth labels if available) to compute the metrics listed above. 
        Libraries like **Scikit-learn** provide functions for all these metrics.


Clustering algorithm: K-means If it is not properly working can be wither that th embeddigns are quite terribles or that the implicit assumptions
                    underlying k-means are not met.
Accuracy scores: Adjusted Mutual Information and Adjusted Rand Index

For visualization porpuses the PCA is used. PCA allows to project the data from the original n-dimensional space into a lower dimensional space. 
Subsequently, we can use PCA to project into a 2/3-dimensional space and plot the data and the clusters in this new space.
'''

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

from sklearn.datasets import make_blobs

def Embedding_Scores(Embeddings,True_Labels,Visible = True):
    
    # ------------ K-MEANS ------------
    
    '''
    In practice, the k-means algorithm is very fast (one of the fastest clustering algorithms available), but it falls in local minima. 
    That’s why it can be useful to restart it several times (ruled by n_init term)
    
    Notes:
        1) It expects embeddings to be already scaled
    
    '''
    n_clusters = len(np.unique(True_Labels))    
    
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0)
    estimator = kmeans.fit(Embeddings)
    
    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results = [m(True_Labels, estimator[-1].labels_) for m in clustering_metrics]
    
    
    
    if Visible == True:
        reduced_data = PCA(n_components=2).fit_transform(Embeddings)
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
        kmeans.fit(reduced_data)
        
                # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
        
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect="auto",
            origin="lower",
        )
        
        plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )
        plt.title(
            "K-means clustering on the embeddings (PCA-reduced data)\n"
            "Centroids are marked with white cross"
        )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        
        
    return results 
                    
 
#%%        
Embeddings, True_Labels = make_blobs(
    n_samples=300,
    n_features=10,
    centers=3,
    cluster_std=1.0,
    random_state=42
)
#%%
results = Embedding_Scores(Embeddings,True_Labels,Visible = True)   
