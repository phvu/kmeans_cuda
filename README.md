kmeans_cuda
===========

CUDA implementation of k-means

The original version of k-means in CUDA was made available by Serban Giuroiu at https://github.com/serban/kmeans.

However Serban used pointer of pointers to represent a 2D matrix, which might not be very convenient in some cases. Moreover in my application, I have the data matrix on the device memory already, and the matrix is stored in column major order (to be used in CUBLAS and other CUDA libraries). Therefore I made some changes to Serban's implementation, concretely:

1. The function now works with column major matrix stored in device memory, and the result is also stored in device memory. This reduces the overhead caused by transposing the matrix in Serban's code, and makes it easier to integrate k-means in other applications.
2. A simple CUDA kernel is added for updating the cluster centroids after each iteration. This reduces the overhead caused by multiple memory transfers at each iteration. However I was lazy and this kernel (called `update_cluster`) has not been well optimized.
3. The `membership` array can be set to `NULL` when calling the function if you don't want to have it in the results.
4. Added a parameter for the maximum number of k-means iterations.

With the new kernel, the program seems to be faster. I already included a simple test case and benchmark in `main.cpp`, you can compile and run it yourself. Serban's original version is also included.
