from multiprocessing.dummy import Pool

# class wrapper:

class KMeans_UR():

    def _default_distance(A,B):

        return np.linalg.norm(A-B)

    def __init__(self,

        k, 
        rand_iters=10, max_iters=100, 
        abs_tol=1e-16, rel_tol=1e-16, 
        dist_func=_default_distance,
        splits = 0):

        self.k = k

        self.rand_iters = rand_iters
        self.max_iters = max_iters

        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        self.dist_func = dist_func
        self.splits = splits #for parallel use

    def fit(self, data):

        # fit the data here, parameters can be easily reuse.

        self.data = data

        # init based on inertia:

        if (self.rand_iters > 0):

            self.centroids_p, self.clusters_p = init_centroids_best_inertia(self.data, self.k, self.rand_iters, self.dist_func)
        else:
            self.centroids_p = init_centroids_random(self.data, self.k)

            # use the pool -> here

            self.clusters_p = centroid_asignment(

                self.data, 
                self.centroids_p, 
                self.dist_func
            )

        # loop the search:        

        self.centroids_p, self.cluster_mask, self.inertia = loop_until_equal_centroids(

            self.data, self.clusters_p, self.centroids_p, 
            self.max_iters, 
            self.abs_tol, self.rel_tol, 
            self.dist_func,
            self.splits
        )

    def centroids(self):

        return self.centroids_p

    def clusters(self):

        return point_labels_to_clusters(self.data, self.cluster_mask, self.k)

    def cluster_tags(self):

        return self.cluster_mask
