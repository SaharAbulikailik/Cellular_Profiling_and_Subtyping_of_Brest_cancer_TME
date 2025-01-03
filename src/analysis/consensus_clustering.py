'''Cluster patient based on their pdf values of a single feature'''
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.cluster import KMeans
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import itertools
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans

# GLOBALS
DATA_DIR = Path('C://Users//sabulikailik//PycharmProjects//researchproject')

# OUTSIDE UTILITIES
def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat,n_clusters,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]

    clusters = fcluster(res_linkage, n_clusters, criterion='maxclust')
    return seriated_dist, res_order, res_linkage, clusters

# FUNCTIONS
def consensus_clustering(df:pd.DataFrame,
                         n_clusters:int,
                         n_iterations:int,
                         percentage:float):
    '''Consensus Clustering based on the supplied DataFrame.

    Distance is measured by square euclidean.
    '''
    # Normalize feature columns
    df_out = df.copy()

    # K-Means
    model = KMeans(n_clusters=n_clusters,n_init=10)
    N = len(df)
    training_sample = int(percentage*N)
    for i in range(n_iterations):
        X = df.sample(training_sample)
        if len(X.values.shape) == 1:
            logits = model.fit_predict(X.values.reshape(-1,1))
        else:
            logits = model.fit_predict(X.values)
        X.loc[:, 'iteration'] = logits+1
        df_out = df_out.join(X['iteration'], rsuffix=str(i))
        centroids = model.cluster_centers_

    return df_out.filter(regex='iteration*', axis=1)

def cluster_similarities(clusters: pd.DataFrame, n_clusters: int):
    '''Create a similarity matrix and cluster after KMeans alg.'''
    # Initialize
    C = clusters.iloc[:, 0].max()
    M = np.ones([len(clusters), len(clusters)])
    T = np.ones([len(clusters), len(clusters)])

    # Cycle through groupings
    for col in clusters.columns:
        # Sampled Together
        tmp_df = clusters.loc[:, col].dropna()
        for x, y in itertools.combinations(tmp_df.index, 2):
            T[x, y] += 1

        # Clustered Together
        groups = clusters.groupby(col)
        for name, group in groups:
            idx = group.index
            for x, y in itertools.combinations(idx, 2):
                M[x, y] += 1

    # Clean and reformat
    M = M / T
    M = np.minimum( M, M.transpose())
    M = 1-M

    # Skip Trivial linkage (1 cluster)
    if C != 1:
        (M_ordered, res_order, res_linkage, clusters) = compute_serial_matrix(M, n_clusters)
    else:
        M_ordered = M


    return M, M_ordered,clusters, res_order

def plot_kaplan_meier(df):
    '''Plot a kaplan meier curve based on patient stratified by 'labels'.'''
    N = len(df['label'].unique())
    fig, ax = plt.subplots(1, 1)
    for l in range(N):
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[df['label']==l, 'Survival (months)'], df.loc[df['label']==l, 'Vital status (1=dead)'], label=l)
        kmf.plot(ax=ax)



def plot_centroids(centroids):
    '''Plot centoids (PDFs) to visualize the difference in clusters.'''
    plt.figure()
    for i, c in enumerate(centroids):
        plt.plot(c, label=f"Centroid {i}")
    plt.legend()



