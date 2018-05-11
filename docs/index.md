# Basic concepts
The spikesorting_tsne project is a new implementation of van der Maaten's t-TSNE algorithm. It has been partially implemented in CUDA so that it now can run significantly faster than the original impelmentation and also allows at least two orders of magnitude more samples. It is a fully rewritten new version of the [t-sne-bhcuda](https://github.com/georgedimitriadis/t_sne_bhcuda) algorithm which it now fully supersedes (this version is faster, allows larger sample sizes, corrects some bugs and has the CUDA code written in python). 

Although the t-SNE algorithm can be used for any matrix of samples x features, the spikesorting-tsne project also has a number of functions available that make easy the use of the t-SNE algorithm with groups of spikes that have been sorted by the kilosort algorithm. Thes functions take the result of the kilosort spikesorting algorithm and produce a matrix of spikes x distance to template that can be run by the t-SNE algorithm. Given the large number of spikes that can now be collected in a single recording some functions are also provided to trim the spike number in a way that is fully representative of the templates (potential single units) that kilosort discovers.

Running t-SNE on the kilosort results (or any other automatic spikesorting algoritm) can be used to create 2D embeddings of spikes seperated in groups that make further manual curation of the spikesorting results easier and intuitive.

## Instalation
### The easy way
The best way to install the package is to install it as a conda package using 'conda install -c georgedimitriadis spikesorting\_tsne' (after you have done 'conda install anaconda-client' to install access to the anaconda cloud where the code is hosted). This will add the Barnes\_Hut.exe executable into the Scripts folder (for Windows) or the bin folder (for Linux) of the python environment that you installed the package in. The t_sne() function in the tsne.py script of the module will call this executable at the appropriate time.

### The hard way
If you want to install the package from code then you need to get two repositories. One is a python repository, [spikesorting\_tsne](https://github.com/georgedimitriadis/spikesorting_tsne) (where this documentation is also hosted). You can either get the repository and put the code anywhere you want or install it as a package from Pypi. The second repository is a C++ repo, [spikesorting\_tsne\_bhpart](https://github.com/georgedimitriadis/spikesorting_tsne_bhpart), that compiles to the Barnes\_Hut.exe. This repository is a Visual Studio 2015 project. If you want to compile the code for Linux then just make a C++ project and copy the code over. The code will compile fine in Linux. The Barnes\_Hut.exe will then have to either be copied over to the Scripts folder (for Windows) or bin folder (for Linux) of your Python installation or its folder can just be passed as an argument (exe_dir=...) to the t_sne() function.

IMPORTANT NOTE:
> Before installing spikesorting\_tsne make sure you have a system that can run numba.cuda functions, i.e. you have a working CUDA installation and a numba installation that works with the GPU (also see below in Requirements).

### Requirements
To run the t-sne part of the code implemented in python you need the following python packages:

1. matplotlib
2. numpy
3. numba >= 0.37.0
4. pyculib >= 1.0.2
5. cython

Numba and pyculib libraries have requirements of their own in how your system should be set up in regards to CUDA installation. You need to follow numba and pyculib documentation so that you are able to utilize the numba.cuda functions and the pyculib functions. This of course is OS dependent. Again, conda is your friend here.

If you install the spikesorting\_tsne package from conda then all the above requirements will be also installed but not the extra requirements for numba.cuda (i.e. CUDA won't magically appear in your system). 

The use of numpy and matplotlib is rather basic and should be fully forwards compatible. Things change with the numba and the pyculib packages every now and then in in ways that break backwards compatibility. If things break in the future reverting to the above mentioned versions should make things work again.

ADVICE: 
>First sort out your numba/CUDA/pyculib environment and then install any way you want the spikesorting\_tsne package!

To run the kilosort helper functions implemented in python you will also need:

1. pandas

The C++ code does not have any requirements.


## Code Architecture
The code is divided into two parts. One that is implemented in python and one in C++.

### The Python codebase
The python code implements three things:

1. The first part of the original t-SNE algorithm that creates the distances of all samples to all other samples. In order to allow very large number of samples (1M plus) the code sorts the distances of each sample to all others and keeps in memory only the smallest perplexity * 3 (together with the indices of those samples). This part of the code is implemented in python using numba and pyculib and utilises the GPU to significantly accelerate the calculation and sorting of the distances.

2. The wrapping of both the above GPU python part and the Barnes\_Hut executable into a single t\_sne() function that properly runs the whole algorithms.

3. A number of helper functions that make it easy to get kilosort's results into the t-SNE algorithm and also subdivide the spikes from a recording into smaller groups that can be handled by the t-SNE algorithm and still represent appropriately the distribution of spikes amongst the found single unit templates

The C++ code implements the actual barnes hut iterations that generate the actual 2D or 3D embedding given the distances of the samples in their high dimensional space.

### Important note for Windows users 
>If you have not used cuda before, then you need to be aware that windows by default will stop and restart the nvidia driver if it thinks that the gpu is stuck. That by default will happen if the gpu does anything that takes longer than 2 seconds. The current code will not work under these conditions with sample sizes over a certain number. If the code requires more than 2 seconds to calculate the distances then windows will restart the driver and the program will fail (you will get a notification of this at the bottom of your screen). In order to get windows off your back do what he says: [Nvidia Display Device Driver Stopped Responding And Has Recovered Successfully (FIX)](https://www.youtube.com/watch?v=QQJ9T0oY-Jk). Also have a look here for MSDN info on the relative registry values [TDR Registry Keys](https://docs.microsoft.com/en-gb/windows-hardware/drivers/display/tdr-registry-keys).