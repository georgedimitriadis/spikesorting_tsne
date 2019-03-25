
import numpy as np
from numba import cuda
import pyculib.blas as cublas
import pyculib.sorting as sorting
import numba
from numba import float32, guvectorize
import math
from . import stopwatch as sw

THREADS_PER_BLOCK = 32


@guvectorize([(float32[:, :], float32[:])], '(m,k)->(m)', nopython=True)
def _create_dot_product(a, dots_a):
    """
    Creates the dot product a[i] . a[i] for each row of the matrix and returns the vector of products

    :param a: The matrix
    :type a: float32[:,:]
    :param dots_a: The resulting vector of dot product squares
    :type dots_a: float32[:]
    """
    for i in np.arange(a.shape[0]):
        dots_a[i] = np.dot(a[i, :], a[i, :])


@cuda.jit
def _sums_of_dots_gpu(dots_a, dots_b, s_o_dots):
    """
    GPU kernel function creating the sums of all values of dots_a vector with all values of dots_b vector

    :param dots_a: The first vector
    :type dots_a: device float32[:]
    :param dots_b: The second vector
    :type dots_b: device float32[:]
    :param s_o_dots: The resulting sums
    :type s_o_dots: device float32[:,:]
    """
    a_index = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x

    if a_index < dots_a.shape[0]:
        for b_index in range(dots_b.shape[0]):
            s_o_dots[a_index, b_index] = dots_a[a_index] + dots_b[b_index]


def _get_inner_products(a, b, verbose):
    """
    Runs the _create_dot_product function on two matrices a and b and times itself

    :param a: The first matrix
    :type a: float32[:, :]
    :param b: The second matrix
    :type b: flaot32[:, :]
    :param verbose: If true print the time the function took to complete
    :type verbose: bool
    :return: The resulting vectors
    :rtype: (float32[:], float32[:])
    """
    with sw.Stopwatch() as stopwatch:
        dots_a = _create_dot_product(a)
        dots_b = _create_dot_product(b)

    if verbose:
        print("Making the dot products Time: {0:.3f} s".format(stopwatch.total_run_time))

    return dots_a, dots_b


def _get_sum_of_dot_products(dots_a, dots_b, distances_on_gpu, verbose):
    """
    Put the dot products on the gpu, calculate the ||a||^2 + ||b||^2 sum of dot products matrix (a.a + b.b) (on the gpu)
    and time the whole function

    :param dots_a: The first vector
    :type dots_a: float32[:]
    :param dots_b: The second vector
    :type dots_b: float32[:]
    :param distances_on_gpu: The resulting per element sums matrix
    :type distances_on_gpu: device float32[:, :]
    :param verbose: If true print the time required to complete the function
    :type verbose: bool
    """
    with sw.Stopwatch() as stopwatch:
        ddots_a = cuda.to_device(np.asfortranarray(dots_a))
        ddots_b = cuda.to_device(np.asfortranarray(dots_b))

        threads_per_block = THREADS_PER_BLOCK
        blocks_per_grid = math.ceil(ddots_a.shape[0] / threads_per_block)
        _sums_of_dots_gpu[blocks_per_grid, threads_per_block](ddots_a, ddots_b, distances_on_gpu)
        numba.cuda.synchronize()

    if verbose:
        print("Summing the dot products on the GPU Time: {0:.3f} s".format(stopwatch.total_run_time))


def _get_gpu_general_matrix_matrix_product(m, n, k, a, b, distances_on_gpu, verbose):
    """
    Calculate the -2<a,b> cross dot products matrix and do the sum (||a||^2 + ||b||^2) -2<a,b>

    :param m: First dimension of first matrix (number of samples in first matrix)
    :type m: int
    :param n: First dimension of second matrix (number of samples in second matrix)
    :type n: int
    :param k: Second dimension of both matrices (number of element in each sample)
    :type k: int
    :param a: First matrix
    :type a: float32[m, k]
    :param b: Second matrix
    :type b: float32[n, k]
    :param distances_on_gpu: Resulting distance matrix
    :type distances_on_gpu: float32[m, n]
    :param verbose: If true print the time the function took to complete
    :type verbose: bool
    """
    with sw.Stopwatch() as stopwatch:
        da = cuda.to_device(np.asfortranarray(a))
        db = cuda.to_device(np.asfortranarray(b))
        blas = cublas.Blas()
        blas.gemm('N', 'T', m, n, k, -2.0, da, db, 1.0, distances_on_gpu)
        numba.cuda.synchronize()

    if verbose:
        print("cuBLAS gemm Time: {0:.3f} s".format(stopwatch.total_run_time))


def _calculate_distances_on_gpu(a, b, distances_on_gpu, verbose=False):
    """
    Calculate the pairwise distances between all row vectors of matrix a and all row vectors of matrix b (on the gpu)

    :param a: First matrix
    :type a: float32[m, k]
    :param b: Second matrix
    :type b: float32[n, k]
    :param distances_on_gpu: The resulting distances between m vectors of matrix one and n vectors of matrix two
    :type distances_on_gpu: device float32[m, n]
    :param verbose: If true print out the times of all steps in this function
    :type verbose: bool
    """
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]

    dots_a, dots_b = _get_inner_products(a, b, verbose)

    _get_sum_of_dot_products(dots_a, dots_b, distances_on_gpu, verbose)

    _get_gpu_general_matrix_matrix_product(m, n, k, a, b, distances_on_gpu, verbose)


def _segment_sort_transposed_distances_get_knns(num_of_neighbours, distances_on_gpu, number_of_sorts,
                                                verbose=False):
    """
    Sorts the distances_on_gpu matrix (on the gpu) and for each element returns the num_of_neighbours smallest
    distances with their position on the vector of elements (indices)

    :param num_of_neighbours: Number of nearest neighbours to each element to return
    :type num_of_neighbours: int
    :param distances_on_gpu: The full, unsorted, pairwise elements distances
    :type distances_on_gpu: device float32[m , m]
    :param number_of_sorts: Given the GPU's memory, in how many pieces must the algorithm break the given distance matrix and sort each piece
    :type number_of_sorts: int
    :param verbose: If true print some info
    :type verbose: bool
    :return: The num_of_neighbors nearest elements' indices and distances vectors
    :rtype: float32[m, k], float32[m, k]
    """
    m = distances_on_gpu.shape[0]  # all spikes
    n = distances_on_gpu.shape[1]  # part of spikes in this distances_on_gpu matrix

    selected_sorted_distances = np.empty((n, num_of_neighbours))
    selected_sorted_indices = np.empty((n, num_of_neighbours))

    with sw.Stopwatch() as stopwatch:
        # Create an array with the indices each iteration will step through
        p = np.append(np.arange(0, n, int(n / number_of_sorts)), n)
        for i in np.arange(1, p.shape[0]):
            # Get how many elemnts each iteration will do
            delta_n = p[i] - p[i - 1]

            # Prepare for sorting in this iteration
            # 1. Get this iteration's keys (distances) onto the host from the gpu and reshape them appropriately
            keys = np.ascontiguousarray(distances_on_gpu.copy_to_host()[:, p[i - 1]:p[i]].transpose().reshape(m * delta_n), dtype=np.float32)
            # 2. Create the values (indices) appropriately reshaped
            values = np.ascontiguousarray(np.tile(np.arange(m), (delta_n, 1)).reshape(m * delta_n), dtype=np.int32)
            # 3. Define the segments
            segments = np.ascontiguousarray(np.arange(m, m * delta_n, m), dtype=np.int32)
            # 4. Do the segmented sort
            sorting.segmented_sort(keys=keys, vals=values, segments=segments)

            # Put the smallest num_of_neighbours distances (and indices) in the selected_sorted_distances
            # (selected_sorted_indices)
            keys = np.reshape(keys, (delta_n, m))[:, :num_of_neighbours]
            values = np.reshape(values, (delta_n, m))[:, :num_of_neighbours]
            selected_sorted_distances[p[i - 1]:p[i], :] = keys[:, :]
            selected_sorted_indices[p[i - 1]:p[i], :] = values[:, :]
            if verbose:
                print('     Sorted {} of {} segments of this iteration'.format(i, p.shape[0] - 1))

    print("SORTING TIME: {0:.3f} s".format(stopwatch.total_run_time))

    return selected_sorted_indices, selected_sorted_distances


def _get_required_gpu_memory():
    """
    Return the number of bytes available in the GPU to do all following steps
    :return: The number of GPU bytes
    :rtype: int
    """
    gpu_mem = cuda.current_context().get_memory_info()
    available_gpu_mem = 0.5 * gpu_mem[0]

    return available_gpu_mem


def calculate_knn_distances(samples_matrix, perplexity=100, verbose=True):
    """
    Calculates the k (perplexity * 3 + 1) nearest neighbors of all row vectors (samples) in the sample_matrix. It uses
    the GPU to do this so it has to take into account the available GPU memory. Given that, it does the comparison in
    steps where the first matrix is the full samples matrix and the second matrix is a sub set of it. Each step first
    calculates all distances between the row vectors of the first and second matrix and then sorts these distances and
    keep the k closest ones.

    :param samples_matrix: The full smaples x dimensions matrix where every sample is a row in the matrix
    :type samples_matrix: float32[m, n]
    :param perplexity: The definition of how many nearest neighbors to keep (k = perplexity * 3 + 1)
    :type perplexity: int
    :param verbose: If true the function prints out the time takes to complete the different parts of it
    :type verbose: bool
    :return: The num_of_neighbors nearest elements' indices and distances vectors
    :rtype: float32[m, k], float32[m, k]
    """
    with sw.Stopwatch() as outside_stopwatch:
        num_of_neighbours = perplexity * 3 + 1  # That is how the original t-sne defined k nearest neighbours
        m = samples_matrix.shape[0]  # number of samples in the samples x dimensions matrix

        closest_indices = np.empty((m, num_of_neighbours))
        closest_distances = np.empty((m, num_of_neighbours))

        available_gpu_mem = _get_required_gpu_memory()

        n = int(np.ceil(available_gpu_mem / (4 * m)))  # The number of samples to compare in each iteration with the full m
        # samples (defined by the GPU's available memory)

        number_of_iters = int(np.ceil(m / n))  # The number of iterations so that all m samples have their k nearest
        # neighbours with all other m - 1 samples calculated

        # Generate the pairs of star and end indices for the samples that will be included in the second matrix at every
        # iteration
        indices_of_second_matrices = [(i, i + n) for i in np.arange(0, number_of_iters * (n - 1), n)]
        indices_of_second_matrices[-1] = (indices_of_second_matrices[-1][0], m)

        first_matrix = np.array(samples_matrix, dtype=np.float32)  # first matrix is the full samples matrix

        # Start the loop over iterations filling up the closest_indices and closest_distances matrices
        for iteration in np.arange(number_of_iters):
            # Fill up the second matrix with the correct part of the whole samples matrix
            second_matrix = np.array(samples_matrix[indices_of_second_matrices[iteration][0]:
                                                    indices_of_second_matrices[iteration][1], :],
                                     dtype=np.float32)

            with sw.Stopwatch() as inside_stopwatch:
                if iteration != 0:
                    del distances_on_gpu
                cuda.current_context().deallocations.clear()

                if verbose:
                    print('LOADING UP THE GPU')
                temp = np.array(np.zeros((m, second_matrix.shape[0]), dtype=np.float32))
                distances_on_gpu = cuda.to_device(np.asfortranarray(temp))

                if verbose:
                    print("Loading matrix time:  {0:.3f} s".format(inside_stopwatch.time_elapsed))

                if verbose:
                    print('ITERATION NUMBER: ' + str(iteration + 1))

                _calculate_distances_on_gpu(a=first_matrix, b=second_matrix, distances_on_gpu=distances_on_gpu,
                                            verbose=verbose)

                available_gpu_mem = _get_required_gpu_memory()

                number_of_sorts = int(np.ceil((16 * n * m) / available_gpu_mem))  # 4 is the bytes per float32,
                # 2 is the two arrays that need to be loaded to gpu, the other factor of 2 is probably a doubling
                # overhead in the algorithm

                if verbose:
                    print('     Number of sorting segments = ' + str(number_of_sorts + 1))

                temp_indices, temp_distances = \
                    _segment_sort_transposed_distances_get_knns(num_of_neighbours=num_of_neighbours,
                                                                distances_on_gpu=distances_on_gpu,
                                                                number_of_sorts=number_of_sorts, verbose=verbose)

                closest_indices[indices_of_second_matrices[iteration][0]: indices_of_second_matrices[iteration][1], :] = \
                    np.ascontiguousarray(temp_indices)
                closest_distances[indices_of_second_matrices[iteration][0]: indices_of_second_matrices[iteration][1], :] = \
                    np.ascontiguousarray(temp_distances)
                if verbose:
                    print('FINISHED CALCULATING ' + str(iteration + 1) + ' OF ' + str(number_of_iters) +
                          ' ITERATIONS')

            if verbose:
                print("Spend Time: {0:.3f} s".format(outside_stopwatch.time_elapsed))

    return closest_indices, np.sqrt(np.abs(closest_distances))


def calculate_knn_distances_for_two_small_matrices(samples_matrix_a, samples_matrix_b, perplexity=10, verbose=True):
    """
    Calculates the k (perplexity * 3 + 1) nearest neighbors of all row vectors (samples) in the sample_matrix_a with all
    rows of the sample_matrix_b. It assumes that the matrices are small enough to fit into GPU memory and does no
    memory checking. It returns the (perplexity * 3 + 1) distances of all the samples of the a matrix with the
    (perplexity * 3 + 1) closest samples of the b matrix

    :param samples_matrix_a: The full smaples x dimensions matrix a where every sample is a row in the matrix
    :type samples_matrix_a: float32[m, n]
    :param samples_matrix_b: The full smaples x dimensions matrix b where every sample is a row in the matrix
    :type samples_matrix_b: float32[m, n]
    :param perplexity: The definition of how many nearest neighbors to keep (k = perplexity * 3 + 1)
    :type perplexity: int
    :param verbose: If true the function prints out the time takes to complete the different parts of it
    :type verbose: bool
    :return: The num_of_neighbors nearest elements' indices and distances vectors
    :rtype: float32[m, k], float32[m, k]
    """
    with sw.Stopwatch() as outside_stopwatch:
        num_of_neighbours = perplexity * 3 + 1  # That is how the original t-sne defined k nearest neighbours
        m = samples_matrix_a.shape[0]  # number of samples in the samples x dimensions matrix
        n = samples_matrix_b.shape[0]

        first_matrix = np.array(samples_matrix_a, dtype=np.float32)  # first matrix is the full samples matrix
        second_matrix = np.array(samples_matrix_b, dtype=np.float32)

        with sw.Stopwatch() as inside_stopwatch:
            cuda.current_context().deallocations.clear()

            if verbose:
                print('LOADING UP THE GPU')
            temp = np.array(np.zeros((m, second_matrix.shape[0]), dtype=np.float32))
            distances_on_gpu = cuda.to_device(np.asfortranarray(temp))

            if verbose:
                print("Loading matrix time:  {0:.3f} s".format(inside_stopwatch.time_elapsed))

            _calculate_distances_on_gpu(a=first_matrix, b=second_matrix, distances_on_gpu=distances_on_gpu,
                                        verbose=verbose)

            available_gpu_mem = _get_required_gpu_memory()

            number_of_sorts = int(np.ceil((16 * n * m) / available_gpu_mem))  # 4 is the bytes per float32,
            # 2 is the two arrays that need to be loaded to gpu, the other factor of 2 is probably a doubling
            # overhead in the algorithm

            if verbose:
                print('     Number of sorting segments = ' + str(number_of_sorts + 1))

            temp_indices, temp_distances = \
                _segment_sort_transposed_distances_get_knns(num_of_neighbours=num_of_neighbours,
                                                            distances_on_gpu=distances_on_gpu,
                                                            number_of_sorts=number_of_sorts, verbose=verbose)

            closest_indices = np.ascontiguousarray(temp_indices)
            closest_distances = np.ascontiguousarray(temp_distances)

        if verbose:
            print("Spend Time: {0:.3f} s".format(outside_stopwatch.time_elapsed))

    return closest_indices, np.sqrt(np.abs(closest_distances))