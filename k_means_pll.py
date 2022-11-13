# implementación en paralelo:

from functools import partial
import functools
from itertools import chain
from multiprocessing import Pool
import multiprocessing
import operator
import random
import numpy as np


def init_centroid_single(datos):

    min_, max_ = np.min(datos, axis=0), np.max(datos, axis=0)

def init_centroids_random(datos, k):
    """ init centroids completely random """

    centroids = []

    min_, max_ = np.min(datos, axis=0), np.max(datos, axis=0)

    centroids = np.array( [list(random.uniform(min_, max_)) for _ in range(k)] )

    # return numpy array:

    return centroids

def init_centroids_best_inertia(datos, k, rand_iters, dist_func):
    """ init centroids random the first time and then choose based on best inertia"""

    prev_loss = np.Infinity

    for _ in range(rand_iters):

        init_centroids = init_centroids_random(datos, k)

        # necesitamos los datos para crear valor aleatorio en el caso len == 0
        init_clusters_labels, loss = centroid_asignment(datos, init_centroids, dist_func)

        # etiquetas (indices) para estos centroides iniciales:
        init_clusters = point_labels_to_clusters(datos, init_clusters_labels, k)

        # utilizamos el loss (inercia) para mejorar la búsqueda de centroides iniciales:
        if (loss < prev_loss):

            # asignamos los de menor inercia
            centroids = init_centroids
            clusters = init_clusters

        prev_loss = loss

    return centroids, clusters

def centroid_coords(cluster):
    """ centroid coordinates based on the clusters points (mean of those points)"""

    if (len(cluster) <= 1) : return np.array([-1])

    # return directly numpy array
    return np.mean(cluster, axis=0)

def centroids_recalculate(clusters):
    """calculate new centroids based on cluster data"""

    return np.array ( [centroid_coords(cluster) for cluster in clusters] )

def centroid_points_dist(point, centroides, dist_func):
    """ distance of concrete data point to each of the centroids """

    return np.array( [dist_func(point, cent) for cent in centroides] )

def centroid_asignment(datos, centroides, dist_func):
    """ returns the list of clustered data to corresponding centroid based on distance """

    label_mask = np.array([])

    loss = 0.0

    for x in datos:

        # for each point calculate its distance to each centroid
        dists = centroid_points_dist(x, centroides, dist_func)

        # select the centroid with minimum distance to the point
        centroid_idx = np.argmin(dists)

        # add point to sorted list # numpy array;
        label_mask = np.append(label_mask, centroid_idx)

        # calculate loss
        loss = loss + np.square(dists[centroid_idx])

    return label_mask, loss

def centroid_asignment_parallel(datos, splits, centroides, dist_func):

    # split data with numpy

    splitted = np.array_split(datos, splits)

    # create pool

    pool = multiprocessing.Pool(splits)

    # create the header needed to work with startmap

    params = []
    for x in range(splits):
        params.append([splitted[x], centroides, dist_func])

    # the result is ordered -> startmap return ordered results
    # independientemente de que se ejecute en paralelo

    result = pool.starmap(centroid_asignment, params)

    pool.close()
    pool.join()

    # cluster-asignment function returns two values, 
    # concatenate and create the needed shape.

    res_masks = [res[0] for res in result]
    res_inert = [res[1] for res in result] # inertias for differnt splits

    # the final inertia is the mean of the partial inertias
    loss = sum(res_inert) / splits 
    # reshape the label mask
    label_mask = np.array(res_masks).reshape(-1)

    return label_mask, loss


def point_labels_to_clusters(datos, label_mask, k):

    """ returns the k clusters with their correspondent data points """

    # use numpy arrays;
    clusters = [np.empty( (0, datos.shape[1] )) for _ in range(k)]

    for p_idx in range(len(datos)):

        cluster_idx = int(label_mask[p_idx])

        point_data = datos[p_idx]

        point_reshaped = np.expand_dims(point_data, axis=0)

        clusters[cluster_idx] = np.append(clusters[cluster_idx], point_reshaped, axis=0)

    return clusters

    # Al final, devuelve la lista con los valores. 
    # Es decir: [[(punto1),(punto7),(punto8)], [...,...]] donde cada sublista 
    # tiene los puntos de ese cluster


def loop_until_equal_centroids(
    
    datos, clusters, centroides, 
    max_iters, abs_tol, rel_tol, 
    dist_func, splits):

    """ loop behaviour, stop based on max_iters, or based on inertia """

    for i in range(max_iters):

        if (splits > 0):

            ### PROCESS TO PARALELL -> cluster asignment

            label_mask, loss = centroid_asignment_parallel(datos, splits, centroides, dist_func)

        else:

            label_mask, loss = centroid_asignment(datos, centroides, dist_func)
        
        # keep with the algorithm

        clusters = point_labels_to_clusters(datos, label_mask, k=len(centroides))

        centroides = centroids_recalculate(clusters)

        for c in range(len(centroides)):
            
            # reasignamos los centroides no seleccionados:
            if np.array_equal( centroides[c] , np.array([-1]) ):

                min_, max_ = np.min(datos, axis=0), np.max(datos, axis=0)
                cent_data = random.uniform(min_, max_)
                centroides[c] = cent_data

        #stop criteria based on the inertia

        if i:

            diff = np.abs(prev_loss - loss)

            if diff < abs_tol and diff / prev_loss < rel_tol:
                break

        prev_loss = loss

    return centroides, label_mask, loss