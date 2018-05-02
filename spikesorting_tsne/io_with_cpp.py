
from os import path
from struct import calcsize, pack, unpack
from platform import system
import numpy as np


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


def find_exe_file():
    """
    Tries to find the Barnes_Hut executable given where it should be residing according to the OS and if this is installed
    as a package

    :return: the path to the executable
    :rtype: string
    """
    exe_dir = 'bin'
    exe_file = 'Barnes_Hut'
    if system() == 'Windows':
        exe_dir = 'Scripts'
        exe_file = 'Barnes_Hut.exe'

    dir_to_exe = None
    current_dir = path.dirname(__file__)
    while dir_to_exe == None:
        if path.isdir(path.join(current_dir, exe_dir)):
            dir_to_exe = path.join(current_dir, exe_dir)
        current_dir = path.dirname(current_dir)
        if len(current_dir) == 3:
            return None
    tsne_path = path.join(dir_to_exe, exe_file)

    return tsne_path


def save_data_for_barneshut(files_dir, sorted_distances, sorted_indices, num_of_dims, perplexity, theta, eta,
                            exageration, iterations, random_seed, verbose):
    """
    Save the pre calculated spike distances and the corresponding spike indices to disk in a file with a header carrying
    all the required paramteres for the barnes hut algorithm to run. The file's name will be data.dat
    This will be read by the barnes hut executable which will run the t-sne loop on the distances and produce the t-sne
    results.

    :param files_dir: the folder where the data.dat file will be saved
    :type files_dir: string
    :param sorted_distances: the distances of the nearest spikes to each spike (sorted)
    :type sorted_distances: float32[:, 3*perplexity +1]
    :param sorted_indices: the indicies of the nearest spikes to each spike (sorted)
    :type sorted_indices: float32[:, 3*perplexity +1]
    :param num_of_dims: how many dimensions should the t-sne embedding have (2 or 3)
    :type num_of_dims: int
    :param perplexity: the number of closest spikes considered are 3 * perplexity
    :type perplexity: int
    :param theta: the angle on the t-sne embedding inside of which all spikes are considered a single point (0 to 1)
    :type theta: float
    :param eta: the learning rate of the algorithm
    :type eta: float
    :param exageration: the initial blow out (push out) of the points on the embedding
    :type exageration: float
    :param iterations: the number of iterations of the t-sne algorithm
    :type iterations: int
    :param random_seed: the random seed to create the initial random positions of the points on the embedding
    :type random_seed: float
    :param verbose: levels of verbosity. 0 is no output. If 3 then the algorithm will also save all intermediate t-sne
    embeddings. Useful for making videos of the t-sne process

    :type verbose: int
    """
    sorted_indices = np.array(sorted_indices, dtype=np.int32)
    num_of_spikes = len(sorted_distances)
    num_of_nns = len(sorted_distances[0])

    filename = 'data.dat'

    with open(path.join(files_dir, filename), 'wb') as data_file:
        # Write the t_sne_bhcuda header
        data_file.write(pack('dddiiiiiii', theta, eta, exageration, num_of_spikes, num_of_dims, num_of_nns, iterations,
                             random_seed, verbose, perplexity))
        # Write the data
        for sample in sorted_distances:
            data_file.write(pack('{}d'.format(len(sample)), *sample))

        for sample in sorted_indices:
            data_file.write(pack('{}i'.format(len(sample)), *sample))


def load_tsne_result(files_dir, filename='result.dat'):
    """
    Load the results of the t-sne executable into a numpy array

    :param files_dir: the folder where the t-sne result file is
    :type files_dir: string
    :param filename: the filename of the results file
    :type filename: string
    :return: the t-sne results
    :rtype: float32[:, :]
    """
    # Read and pass on the results
    with open(path.join(files_dir, filename), 'rb') as output_file:
        # The first two integers are the number of samples and the dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)
        # Collect the results, but they may be out of order
        results = [_read_unpack('{}d'.format(result_dims), output_file) for _ in range(result_samples)]

        return np.array(results)


def load_barneshut_data(files_dir, filename='data.dat', data_has_exageration=True):
    """
    Load the spike distances and spike indices into numpy array from the data.dat file previously generated. Using this
    one can get the spike distances and indices and then resave them to a new data.dat file with a different parameters
    header to run t-sne again with different parameters.

    :param files_dir: the folder where the data file is
    :type files_dir: string
    :param filename: the name of the data file
    :type filename: string
    :param data_has_exageration: for backwards compatibility. old data files did not have the exageration parameter
    :type data_has_exageration: bool
    :return: the sorted distances of the closest spikes to each spike, the sorted indices of the closest spikes to each
    spike, the dictionary of the parameters used in this data file

    :rtype: float32[:,:], float32[:,:], dict
    """
    data_file = path.join(files_dir, filename)

    print('Loading previously calculated high dimensional distances')

    with open(data_file, 'rb') as output_file:
        if data_has_exageration:
            theta, eta, exageration, num_of_spikes, num_of_dims, num_of_nns, iterations, \
            random_seed, verbose, perplexity = _read_unpack('dddiiiiiii', output_file)

            parameters_dict = {'theta': theta, 'eta': eta, 'exageration': exageration,  'num_of_spikes': num_of_spikes,
                               'num_of_dims': num_of_dims, 'num_of_nns': num_of_nns, 'iterations': iterations,
                               'random_seed': random_seed, 'verbose': verbose, 'perplexity': perplexity}

        else:
            theta, eta, num_of_spikes, num_of_dims, num_of_nns, iterations, \
            random_seed, verbose, perplexity = _read_unpack('ddiiiiiii', output_file)

            parameters_dict = {'theta': theta, 'eta': eta, 'num_of_spikes': num_of_spikes,
                               'num_of_dims': num_of_dims, 'num_of_nns': num_of_nns, 'iterations': iterations,
                               'random_seed': random_seed, 'verbose': verbose, 'perplexity': perplexity}

        sorted_distances = np.array(
            [_read_unpack('{}d'.format(num_of_nns), output_file) for _ in range(num_of_spikes)])

        sorted_indices = np.array(
            [_read_unpack('{}i'.format(num_of_nns), output_file) for _ in range(num_of_spikes)])

    print('     Size of distances matrix: ' + str(sorted_distances.shape))
    print('     Size of indices matrix: ' + str(sorted_indices.shape))

    return sorted_distances, sorted_indices, parameters_dict

