# Basic concepts
The spikesorting_tsne project is a new implementation of van der Maaten's t-TSNE algorithm. It has been partially implemented in CUDA so that it now can run significantly faster than the original impelmentation and also allows at least two orders of magnitude more samples. It is a fully rewritten new version of the [t-sne-bhcuda](https://github.com/georgedimitriadis/t_sne_bhcuda) algorithm which it now fully supersedes (this version is faster, allows larger sample sizes, corrects some bugs and has the CUDA code written in python). 

Although the t-SNE algorithm can be used for any matrix of samples x features, the spikesorting-tsne project also has a number of functions available that make easy the use of the t-SNE algorithm with groups of spikes that have been sorted by the kilosort algorithm. Thes functions take the result of the kilosort spikesorting algorithm and produce a matrix of spikes x distance to template that can be run by the t-SNE algorithm. Given the large number of spikes that can now be collected in a single recording some functions are also provided to trim the spike number in a way that is fully representative of the templates (potential single units) that kilosort discovers.

Running t-SNE on the kilosort results (or any other automatic spikesorting algoritm) can be used to create 2D embeddings of spikes seperated in groups that make further manual curation of the spikesorting results easier and intuitive.

## Code Architecture
The code is divided into two parts. One that is implemented in python and one in C++.