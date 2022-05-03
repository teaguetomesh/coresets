# coresets

# Coreset Clustering on Small Quantum Computers
This code was used to generate the data and plots in the paper: https://doi.org/10.3390/electronics10141690. This work explores the use of coresets - small sets of data which accurately summarize a larger data set - to fit large machine learning problems onto small quantum computers.

The bulk on the code is contained in the `kMeans` directory. The benchmark data sets used in the paper are contained in the `Datasets` directory stored as binary `.npy` files (see https://numpy.org/doc/stable/reference/generated/numpy.load.html).

The code for generating the coresets studied in the paper are contained in `coreset.py`.
- *Practical Coreset Constructions for Machine Learning* https://arxiv.org/pdf/1703.06476.pdf
- *New Frameworks for Offline and Streaming Coreset Constructions* https://arxiv.org/pdf/1612.00889.pdf

The file `kmeans_qaoa.py` contains all of the functions used to solve the *k*-means clustering problem using the Quantum Approximate Optimization Algorithm (QAOA). The functions within this file load in and plot the coreset points, construct and optimize the variational QAOA circuit, and execute them on real IBMQ hardware. The jupyter notebook `qaoa.ipynb` is a useful demonstration of these functions.

The `catdog.ipynb`, `cifar10.ipynb`, `epilepsy.ipynb`, `pulsar.ipynb`, `synthetic.ipynb`, `val2017.ipynb`, and `yeast.ipynb` notebooks were used to generate the coreset comparison results in the paper.
