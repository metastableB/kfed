import numpy as np
import scipy
import scipy.sparse as sps
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances as sparse_cdist
from sklearn.utils.extmath import randomized_svd
from kmeans_pytorch import kmeans as torch_kmeans
import torch


def distance_to_set(A, S, sparse=False):
    '''
    S is a list of points. Distance to set is the minimum distance of $x$ to
    points in $S$. In this case, this is computed for each row of $A$.  Note
    that this works with sparse matrices. To enable this behaviour, set
    sparse=True.

    Returns a single array of length len(A) containing corresponding distances.
    '''
    n, d = A.shape
    assert S.ndim == 2
    assert S.shape[1] == d, S.shape[1]
    assert A.shape[1] == d
    assert A.ndim == 2
    # Pair wise distances
    if sparse is False:
        pd = scipy.spatial.distance.cdist(A, S, metric='euclidean')
    else:
        pd = sparse_cdist(A, S)
    assert np.allclose(pd.shape, [A.shape[0], len(S)])
    dx = np.min(pd, axis=1)
    assert len(dx) == A.shape[0]
    assert dx.ndim == 1
    return dx


def get_clustering(A, centers, sparse=False):
    '''
    Returns a list of integers of length len(A). Each integer is an index which
    tells us the cluster A[i] belongs to. A[i] is assigned to the closest
    center.
    '''
    # Pair wise distances
    if sparse is False:
        pd = scipy.spatial.distance.cdist(A, centers, metric='euclidean')
    else:
        pd = sparse_cdist(A, centers)
    assert np.allclose(pd.shape, [A.shape[0], len(centers)])
    indices = np.argmin(pd, axis=1)
    assert len(indices) == A.shape[0]
    return np.array(indices)


def kmeans_cost(A, centers, sparse=False, remean=False):
    '''
    Computes the k means cost of rows of $A$ when assigned to the nearest
    centers in centers.

    remean: If remean is set to True, then the kmeans cost is computed with
    respect to the actual means of the clusters and not necessarily the centers
    provided in centers argument (which might not be actual mean of the
    clustering assignment).
    '''
    clustering = get_clustering(A, centers, sparse=sparse)
    cost = 0
    if remean is True:
        # We recompute mean based on assignment.
        centers2 = []
        for clusterid in np.unique(clustering):
            points = A[clustering == clusterid]
            centers2.append(np.mean(points, axis=0))
        centers = np.array(centers2)
    for clusterid in np.unique(clustering):
        points = A[clustering == clusterid]
        dist = distance_to_set(points, centers, sparse=sparse)
        cost += np.mean(dist ** 2)
    return cost


def create_device_data(x, y, dev_k, frac_per_cluster, sparse=False,
                       return_segmatrix=False):
    '''

    Data in $x$ is divided into segments based on the labelling in $y$.  Note
    that $y$ need not be the labelling in the 'classification' sense. It could
    be whatever metric you want segmentation to be based on. The segmentation
    will then be used to ensures that data from no more than dev_k clusters end
    up on each device  and no more than frac_per_cluster fraction of data from
    each cluster (m_0 in main paper).

    Note: The arguments, (dev_k, frac_split), determine the number of devices.
    For instance if dev_k = num_segments, then each device will have data from
    all $k$ clusters. Now if frac_split = 0.1. Then each device can have no
    more than 0.1 fraction of the data from one cluster. Thus there will be at
    least 10 devices. This behaviour is a side effect of how we create device
    data from real data and has nothing to do with our clustering algorithm
    itself --- which can work with any device level split.

    Note: There could be left over data. This is collected as an extra device
    and appended at the end. Note that this device must have fewer than dev_k
    data per device.

    Supports dense and sparse x.

    return_segmatrix: The final device is constructed by just
    concatenating whatever data was left over, as represented by 1s in the
    final segmatrix. This method returns that segmatrix which could be helpful
    in analysis.

    Algorithm:
        First divide the data into segments/clusters. Construct what we call a
        segment matrix (a binary matrix) C. Here C[i, j] indexes segment i and
        chunk j (explained below).

        Now divide the data in each segment (think cluster) to some chunks. We
        will devide chunks among devices such that each device only receives
        one chunk of data. Since the  ber of chunks in different segments are
        different, the row size will be max_number of segments. Set all index
        pairs that lead to valid chunk of a segment to 1 and others to 0.

        Then sample $k'$ 1s from this matrix and set them to 0 afterwards.
        Repeat.

    Note that, due to the way we are sampling, certain devices might get
    multiple chunks from the same segment. We let this be to be true to the
    theory.

    To get a random split, use a list of random segments (i.e. provide random
    clustering based segments as inputs).
    '''
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == len(y)
    # Divide data into devices with (park_k, frac_split) parameters.
    segment_list = []
    for label in np.unique(y):
        a = x[y == label]
        segment_list.append(a)
    # Now that we have the segments (clusters) we create chunks from each
    # cluster and distribute them among the devices.
    assert 0 < frac_per_cluster <= 1
    num_segments = len(segment_list)
    chunks = []
    max_num_chunk = 0
    for segment in segment_list:
        chunk_len = int(frac_per_cluster * segment.shape[0])
        if chunk_len == 0:
            chunk_len = segment.shape[0]
        starts = [start for start in range(0, segment.shape[0], chunk_len)]
        chunk_seg = [segment[x:x+chunk_len] for x in starts]
        max_num_chunk = max(max_num_chunk, len(chunk_seg))
        chunks.append(chunk_seg)

    segmatrix = np.zeros((num_segments, max_num_chunk))
    for i in range(num_segments):
        for j in range(len(chunks[i])):
            segmatrix[i, j] = 1
    # Just some sanity checks
    valid_indices = np.argwhere(np.sum(segmatrix, axis=1) > 0)
    assert len(valid_indices) == num_segments

    device_list = []
    while True:
        valid_indices = np.argwhere(np.sum(segmatrix, axis=1) > 0)
        valid_indices = np.reshape(valid_indices, -1)
        if len(valid_indices) < dev_k:
            break
        choice = [x for x in range(len(valid_indices))]
        choice = np.random.choice(choice, dev_k, replace=False)
        chosen_indices = valid_indices[choice]
        assert len(chosen_indices) == dev_k
        dev = []
        for i in chosen_indices:
            # Pick any segment. We choose argmax based.
            j = np.argmax(segmatrix[i])
            assert segmatrix[i, j] == 1, 'Internal error: Something is wrong'
            assert chunks[i][j].shape[0] > 0, (i, j)
            dev.append(chunks[i][j])
            segmatrix[i, j] = 0
        device_list.append(dev)

    # Handle the case of left over data
    valid_indices = np.argwhere(np.sum(segmatrix, axis=1) > 0)
    valid_indices = np.reshape(valid_indices, -1)
    if len(valid_indices) > 0:
        assert len(valid_indices) < dev_k
        dev = []
        rem_chunks = np.argwhere(segmatrix > 0)
        for (i, j) in rem_chunks:
            dev.append(chunks[i][j])
        device_list.append(dev)
    # Merge data on each device together
    ret = []
    for device in device_list:
        if sparse:
            a = sps.vstack(device)
        else:
            a = np.concatenate(device)
        ret.append(a)
    if return_segmatrix:
        # In case the user wants the final seg matrix of the last cluster.
        ret = (ret, segmatrix)
    return ret


def kmeans_pp(A, k, weighted=True, sparse=False, verbose=False):
    '''
    Returns $k$ initial centers based on the k-means++ initialization scheme.
    With weighted set to True, we have the standard algorithm. When weighted is
    set to False, instead of picking points based on the D^2 distribution, we
    pick the farthest point from the set.

    A is nxd data matrix (sparse or dense). k is the number of clusters.

    Returns a (k x d) dense matrix.

    K-means ++
    ----------
     1. Choose one center uniformly at random among the data points.
     2. For each data point x, compute D(x), the distance between x and
        the nearest center that has already been chosen.
     3. Choose one new data point at random as a new center, using a
        weighted probability distribution where a point x is chosen with
        probability proportional to D(x)2.
     4. Repeat Steps 2 and 3 until k centers have been chosen.
    '''
    n, d = A.shape
    if n <= k:
        if sparse:
            A = A.toarray()
        return np.aray(A)
    index = np.random.choice(n)
    if sparse is True:
        B = np.squeeze(A[index].toarray())
        assert len(B) == d
        inits = [B]
    else:
        inits = [A[index]]
    indices = [index]
    t = [x for x in range(A.shape[0])]
    distance_matrix = distance_to_set(A, np.array(inits), sparse=sparse)
    distance_matrix = np.expand_dims(distance_matrix, axis=1)
    while len(inits) < k:
        if verbose:
            print('\rCenter: %3d/%4d' % (len(inits) + 1, k), end='')
        # Instead of using distance to set we can compute this incrementally.
        dx = np.min(distance_matrix, axis=1)
        assert dx.ndim == 1
        assert len(dx) == n
        dx = dx**2/np.sum(dx**2)
        if weighted:
            choice = np.random.choice(t, 1, p=dx)[0]
        else:
            choice = np.argmax(dx)
        if choice in indices:
            continue
        if sparse:
            B = np.squeeze(A[choice].toarray())
            assert len(B) == d
        else:
            B = A[choice]
        inits.append(B)
        indices.append(choice)
        last_center = np.expand_dims(B, axis=0)
        assert last_center.ndim == 2
        assert last_center.shape[0] == 1
        assert last_center.shape[1] == d
        dx = distance_to_set(A, last_center, sparse=sparse)
        assert dx.ndim == 1
        assert len(dx) == n
        dx = np.expand_dims(dx, axis=1)
        a = [distance_matrix, dx]
        distance_matrix = np.concatenate(a, axis=1)
    if verbose:
        print()
    return np.array(inits)


def awasthisheffet(A, k, useSKLearn=True, sparse=False):
    '''
    This is the algorithm from Pranjal Awasthi and Or Sheffet's paper.  The
    implementation here uses kmeans++ to get initial centers (\nu in the paper)
    instead of using a 10-approx algorithm.

    1. Project onto $k$ dimensional space.
    2. Use $k$-means++ to initialize.
    3. Use 1:3 distance split to improve initialization.
    4. Run Lloyd steps and return final solution.

    Returns a sklearn.cluster.Kmeans object with the clustering information and
    the list $S_r$.
    '''
    assert A.ndim == 2
    n = A.shape[0]
    d = A.shape[1]
    # If we don't have $k$ points then return the matrix as its the best $k$
    # partition trivially.
    if n <= k:
        if sparse:
            A = np.array(A.toarray())
        return A, None
    # This works with sparse and dense matrices. Returns dense always.
    U, Sigma, V = randomized_svd(A, n_components=k, random_state=None)
    # Columns of $V$ are eigen vectors
    V = V.T[:, :k]
    # Sparse and dense compatible. A_hat is always dense.
    A_hat = A.dot(V)
    # For simplicity we use k-means++ locally. Its log-k competitive
    # --- our theorems follow through with any $t$ competitive algorithm with
    # and extra t-factor. Slight hand-waving here, but moral points are valid.
    # More importantly, this is more practical.
    inits = kmeans_pp(A_hat, k, sparse=False)
    # Run STEP 2, modified Lloyd. We have vectorized it for speed up.
    if sparse is False:
        pd = scipy.spatial.distance.cdist(inits, A_hat)
    else:
        pd = sparse_cdist(inits, A_hat)
    Sr_list = []
    for r in range(k):
        th = 3 * pd[r, :]
        remaining_dist = pd[np.arange(k) != r]
        assert np.allclose(remaining_dist.shape, [k- 1, n])
        indicator = (remaining_dist - th) < 0
        indicator = np.sum(indicator.astype(int), axis=0)
        assert len(indicator) == n
        # places where indicator is 0 is our set
        Sr = [i for i in range(len(indicator)) if indicator[i] == 0]
        assert len(Sr) >= 0
        Sr_list.append(Sr)
    # We don't mind lloyd_init being dense. Its only k x d.
    lloyd_init = np.array([np.mean(A_hat[Sr], axis=0) for Sr in Sr_list])
    assert np.allclose(lloyd_init.shape, [k, k])
    # Project back to d dimensional space
    lloyd_init = np.matmul(lloyd_init, V.T)
    assert np.allclose(lloyd_init.shape, [k, d])
    # Run Lloyd's method
    if useSKLearn:
        # Works with sparse matrices as well.
        kmeans = KMeans(n_clusters=k, init=lloyd_init)
        kmeans.fit(A)
        ret = (kmeans.cluster_centers_, kmeans)
    else:
        raise NotImplementedError("This is not implemented/tested")
        if not torch.cuda.is_available():
            raise Exception("Cuda is not available.")
        # device = torch.device('cuda:0')
        Atorch = torch.from_numpy(A)
        print("Using pytorch implementation")
        # We use the GPU version from torch
        _, cluster_centers = torch_kmeans(X=Atorch, num_clusters=k,
                                          device='cuda:0')
        ret = (cluster_centers, None)
    return ret


def kfed(x_dev, dev_k, k, useSKLearn=True, sparse=False):
    '''
    The full decentralized algorithm.

    Warning: Synchronous version, no parallelization across devices. Since the
    sklearn k means routine is itself parallel. TODO: Think about
    parallelization based on resource uitlization.

    x_dev: [Number of devices, Data len, dimension]
    dev_k: Integer. The value $k'$ in the paper. Number of clusters
        per device. We use constant for all devices.

    Good explanation of python multi-processing for future parallelization
    benefits.
    https://further-reading.net/2017/01/quick-tutorial-python-multiprocessing/

    Returns: Local estimators (local centers), central-centers
    '''
    def cleaup_max(local_estimators, k, dev_k, useSKLearn=True, sparse=False):
        '''
        Central cleanup phase based on the max-from-set rule.
        '''
        assert local_estimators.ndim == 2
        # The first dev_k points definitely in different target clusters.
        init_centers = local_estimators[:dev_k, :]
        remaining_data = local_estimators[dev_k:, :]
        # For the remaining initialization, use max rule.
        while len(init_centers) < k:
            distances = distance_to_set(remaining_data, np.array(init_centers),
                                        sparse=sparse)
            candidate_index = np.argmax(distances)
            candidate = remaining_data[candidate_index:candidate_index+1, :]
            # Combine with init_centers
            init_centers = np.append(init_centers, candidate, axis=0)
            # Remove from remaining_data
            remaining_data = np.delete(remaining_data, candidate_index, axis=0)

        assert len(init_centers) == k
        # Perform final clustering.
        if useSKLearn:
            # Works with sparse matrices as well.
            kmeans = KMeans(n_clusters=k, init=init_centers)
            kmeans.fit(local_estimators)
            ret = (kmeans.cluster_centers_, kmeans)
        else:
            raise NotImplementedError("This is not implemented/tested")
        return ret

    num_dev = len(x_dev)
    msg = "Not enough devices "
    msg += "(num_dev=%d, dev_k=%d, k=%d)" % (num_dev, dev_k, k)
    assert dev_k * num_dev >= k, msg
    # Run local $k$-means
    local_clusters = []
    for dev in x_dev:
        cluster_centers, _ = awasthisheffet(dev, dev_k, useSKLearn=useSKLearn,
                                            sparse=sparse)
        local_clusters.append(cluster_centers)
    # This is alwasys dense.
    local_estimates = np.concatenate(local_clusters, axis=0)
    msg = "Not enough estimators. "
    msg += "Estimator matrix size: " + str(local_estimates.shape) + ", while "
    msg += "k = %d" % k
    assert local_estimates.shape[0] > k, msg
    # Local estimators are dense
    centers, kmeansobj = cleaup_max(local_estimates, k, dev_k,
                                    useSKLearn=useSKLearn, sparse=False)
    return local_estimates, centers

