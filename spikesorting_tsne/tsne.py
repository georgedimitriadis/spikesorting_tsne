

from . import io_with_cpp as io
from . import gpu
import matplotlib.pylab as pylab
from subprocess import Popen, PIPE
import sys
from os import path
from platform import system


def get_barnes_hut_executable(exe_dir=None):
    """
    Finds the directory that the Barnes_Hut.exe file would be in if this was installed as a package (or uses the user
    defined folder)

    :param exe_dir: the user defined folder of where the Barnes_Hut.exe file is
    :type exe_dir: string
    :return: the folder where the Barnes_Hut.exe file is
    :rtype: string
    """
    if exe_dir is None:
        exe_file = io.find_exe_file()
        if exe_file is None:
            print('Cannot find Barnes_Hut.exe. Please provide a path to it yourself by setting the exe_dir parameter.')
            return
    else:
        if system() == 'Windows':
            exe_file = path.join(exe_dir, 'Barnes_Hut.exe')
        else:
            exe_file = path.join(exe_dir, 'Barnes_Hut.out')

    return exe_file


def run_executable(exe_file, files_dir, verbose):
    """
    Runs the Barnes_Hut.exe executable

    :param exe_file: the full path of the Barnes_Hut executable
    :type exe_file: string
    :param files_dir: the folder where the data.dat file is that the executable is going to read
    :type files_dir: string
    :param verbose: level of verbosity
    :type verbose: int
    """
    with Popen(['Barnes_Hut.exe', ], executable=exe_file, cwd=files_dir, stdout=PIPE, bufsize=1,
               universal_newlines=True) \
            as t_sne_bhcuda_p:
        for line in iter(t_sne_bhcuda_p.stdout):
            print(line, end='')
            sys.stdout.flush()
        t_sne_bhcuda_p.wait()
    assert not t_sne_bhcuda_p.returncode, ('ERROR: Call to Barnes_Hut exited '
                                           'with a non-zero return code exit status, please ' +
                                           ('enable verbose mode and ' if not verbose else '') +
                                           'refer to the t_sne output for further details')


def t_sne(samples, files_dir=None, exe_dir=None, num_dims=2, perplexity=100, theta=0.4, eta=200, exageration=12.0,
          iterations=1000, random_seed=-1, verbose=2):
    """
    Runs the t-SNE algorithm on the samples matrix

    :param samples: the samples x features matrix to run t-SNE on
    :type samples: float32[:,:]
    :param files_dir: where the data.dat file is and where the t-SNE will save its result
    :type files_dir: string
    :param exe_dir: where the Barnes_Hut.exe is (if None then the algorithm will try and find it itself)
    :type exe_dir: string
    :param num_dims: how many dimensions should the t-sne embedding have (2 or 3)
    :type num_dims: int
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
    :return: the t-NSE result
    :rtype: float32[:,num_of_dims]
    """
    data = pylab.demean(samples, axis=0)
    data /= data.max()
    closest_indices_in_hd, closest_distances_in_hd = \
        gpu.calculate_knn_distances(samples_matrix=data, perplexity=perplexity, verbose=True)

    io.save_data_for_barneshut(files_dir, closest_distances_in_hd, closest_indices_in_hd, num_of_dims=num_dims,
                               perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                               iterations=iterations, random_seed=random_seed, verbose=verbose)

    del samples

    exe_file = get_barnes_hut_executable(exe_dir)

    run_executable(exe_file, files_dir, verbose)

    tsne = io.load_tsne_result(files_dir)

    return tsne


def t_sne_from_existing_distances(files_dir, data_has_exageration=True,exe_dir=None, num_dims=2, theta=0.4, eta=200,
                                  exageration=12.0, iterations=1000, random_seed=-1, verbose=2):
    """
    Runs the t-SNE algorithm using a precalculated set of distances (saved in the data.dat file in the files_dir folder)
    This will extract the distances and indices from the data.dat file and will overwrite it with the new parameters
    like num_dims, iterations, etc.. Given that the number of the kept distances is defined by perplexity, this cannot
    be reset but stays the same as used originally to calculate the high dimensional distances.

    :param files_dir: where the data.dat file is and where the t-SNE will save its result
    :type files_dir: string
    :param exe_dir: where the Barnes_Hut.exe is (if None then the algorithm will try and find it itself)
    :type exe_dir: string
    :param num_dims: how many dimensions should the t-sne embedding have (2 or 3)
    :type num_dims: int
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
    :return: the t-NSE result
    :rtype: float32[:,num_of_dims]
    """
    closest_distances_in_hd, closest_indices_in_hd, parameters_dict = io.load_barneshut_data(files_dir,
                                                                                             data_has_exageration=
                                                                                             data_has_exageration)

    perplexity = parameters_dict['perplexity']

    io.save_data_for_barneshut(files_dir, closest_distances_in_hd, closest_indices_in_hd, num_of_dims=num_dims,
                               perplexity=perplexity, theta=theta, eta=eta, exageration=exageration,
                               iterations=iterations, random_seed=random_seed, verbose=verbose)

    exe_file = get_barnes_hut_executable(exe_dir)

    run_executable(exe_file, files_dir, verbose)

    tsne = io.load_tsne_result(files_dir)

    return tsne

