/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         cuda_kmeans.cu  (CUDA version)                            */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
// Copyright (c) 2013 Vu Pham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "kMeansCuda.h"

namespace cuda 
{

void get_kernel_config_given_ratios(int sz1, int sz2, dim3& szGrid, dim3& szBlock
                , int& rowPerThread, int& colPerThread
                , int nThreadXRatio, int nThreadYRatio)
{
    szBlock.x = std::min(sz1, nThreadXRatio);
    szBlock.y = std::min(sz2, nThreadYRatio);
    szBlock.z = 1;
    szGrid.x = szGrid.y = szGrid.z = 1;
    colPerThread = rowPerThread = 1;
    
    if (sz1 > nThreadXRatio || sz2 > nThreadYRatio)
    {
        int ratio = sz1/nThreadXRatio, k;
        for (k = 1; (1 << k) <= ratio; ++k)
        {
            rowPerThread = (2 << (k/2));
        }
        //rowPerThread = 2 << (int)(std::log(std::sqrt((float)sz1/nThreadX))/std::log((float)2));
        szGrid.x = (sz1 + szBlock.x*rowPerThread - 1) / (szBlock.x*rowPerThread);

        ratio = sz2/nThreadYRatio;
        for (k = 1; (1 << k) <= ratio; ++k)
        {
            colPerThread = (2 << (k/2));
        }
        //colPerThread = 2 << (int)(std::log(std::sqrt((float)sz2/nThreadY))/std::log((float)2));
        szGrid.y = (sz2 + szBlock.y*colPerThread - 1) / (szBlock.y*colPerThread);
    }
    assert(szGrid.x*szBlock.x*rowPerThread >= sz1);
    assert(szGrid.y*szBlock.y*colPerThread >= sz2);
}

void get_kernel_config(int sz1, int sz2, dim3& szGrid, dim3& szBlock
                    , int& rowPerThread, int& colPerThread)
{
    // CUDA 2.x: maximum 1024 threads/block. CUDA < 2.x: 512 threads/block
    
    int nThreadX, nThreadY;
    if (sz1 / sz2 >= 2)
    {
        nThreadX = 64; nThreadY = 16;
    }
    else if (sz2 / sz1 >= 2)
    {
        nThreadX = 16; nThreadY = 64;
    }
    else
    {
        nThreadX = nThreadY = 32;
    }
    get_kernel_config_given_ratios(sz1, sz2, szGrid, szBlock
            , rowPerThread, colPerThread, nThreadX, nThreadY);
}

/******************************************************************************/


static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

    return ++n;
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static
float euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;

    for (i = 0; i < numCoords; i++) {
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block. See numThreadsPerClusterBlock in cuda_kmeans().
    unsigned char *membershipChanged = (unsigned char *)sharedMemory;
#if BLOCK_SHARED_MEM_OPTIMIZATION
    float *clusters = (float *)(sharedMemory + blockDim.x);
#else
    float *clusters = deviceClusters;
#endif

    membershipChanged[threadIdx.x] = 0;

#if BLOCK_SHARED_MEM_OPTIMIZATION
    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates! For reference, a Tesla C1060 has 16
    //  KiB of shared memory per block, and a GeForce GTX 480 has 48 KiB of
    //  shared memory per block.
    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();
#endif

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < numObjs) {
        int   index, i;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);

        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (membership[objectId] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

        /* assign the membership to object objectId */
        membership[objectId] = index;

        __syncthreads();    //  For membershipChanged[]

        //  blockDim.x *must* be a power of two!
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] +=
                    membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            intermediates[blockIdx.x] = membershipChanged[0];
        }
    }
}

__global__ static
void compute_delta(int *deviceIntermediates,
                   int numIntermediates,    //  The actual number of intermediates
                   int numIntermediates2)   //  The next power of two
{
    //  The number of elements in this array should be equal to
    //  numIntermediates2, the number of threads launched. It *must* be a power
    //  of two!
    extern __shared__ unsigned int intermediates[];

    //  Copy global intermediate values into shared memory.
    intermediates[threadIdx.x] =
        (threadIdx.x < numIntermediates) ? deviceIntermediates[threadIdx.x] : 0;

    __syncthreads();

    //  numIntermediates2 *must* be a power of two!
    for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        deviceIntermediates[0] = intermediates[0];
    }
}

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)


/*----< cuda_kmeans() >-------------------------------------------------------*/
//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */
float** kMeansHost(float **objects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations)
{
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  **dimObjects;
    float  **clusters;       /* out: [numClusters][numCoords] */
    float  **dimClusters;
    float  **newClusters;    /* [numCoords][numClusters] */

    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership;
    int *deviceIntermediates;

    //  Copy objects given in [numObjs][numCoords] layout to new
    //  [numCoords][numObjs] layout
    malloc2D(dimObjects, numCoords, numObjs, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numObjs; j++) {
            dimObjects[i][j] = objects[j][i];
        }
    }

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    malloc2D(dimClusters, numCoords, numClusters, float);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    malloc2D(newClusters, numCoords, numClusters, float);
    memset(newClusters[0], 0, numCoords * numClusters * sizeof(float));

    //  To support reduction, numThreadsPerClusterBlock *must* be a power of
    //  two, and it *must* be no larger than the number of bits that will
    //  fit into an unsigned char, the type used to keep track of membership
    //  changes in the kernel.
    const unsigned int numThreadsPerClusterBlock = 128;
    const unsigned int numClusterBlocks =
        (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
#if BLOCK_SHARED_MEM_OPTIMIZATION
    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char) +
        numClusters * numCoords * sizeof(float);

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        err("WARNING: Your CUDA hardware has insufficient block shared memory. "
            "You need to recompile with BLOCK_SHARED_MEM_OPTIMIZATION=0. "
            "See the README for details.\n");
    }
#else
    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char);
#endif

    const unsigned int numReductionThreads =
        nextPowerOfTwo(numClusterBlocks);
    const unsigned int reductionBlockSharedDataSize =
        numReductionThreads * sizeof(unsigned int);

    CHECK_CUDA(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));

    CHECK_CUDA(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));

    do {
        CHECK_CUDA(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));

        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        compute_delta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>
            (deviceIntermediates, numClusterBlocks, numReductionThreads);

        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        int d;
        CHECK_CUDA(cudaMemcpy(&d, deviceIntermediates,
                  sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d;

        CHECK_CUDA(cudaMemcpy(membership, deviceMembership,
                  numObjs*sizeof(int), cudaMemcpyDeviceToHost));

        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = membership[i];

            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[j][index] += objects[i][j];
        }

        //  TODO: Flip the nesting order
        //  TODO: Change layout of newClusters to [numClusters][numCoords]
        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;

    /* allocate a 2D space for returning variable clusters[] (coordinates
       of cluster centers) */
    malloc2D(clusters, numClusters, numCoords, float);
    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < numCoords; j++) {
            clusters[i][j] = dimClusters[j][i];
        }
    }

    CHECK_CUDA(cudaFree(deviceObjects));
    CHECK_CUDA(cudaFree(deviceClusters));
    CHECK_CUDA(cudaFree(deviceMembership));
    CHECK_CUDA(cudaFree(deviceIntermediates));

    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return clusters;
}

/******************************************************************************/

__global__ static
void update_cluster(const float* objects, const int* membership, float* clusters
                    , const int nCoords, const int nObjs, const int nClusters
                    , const int rowPerThread, const int colPerThread)
{
    for (int cIdx = 0; cIdx < colPerThread; ++cIdx)
    {
        int c = cIdx * gridDim.y * blockDim.y + blockIdx.y * blockDim.y + threadIdx.y;
        if (c >= nClusters)
            break;
        
        for (int rIdx = 0; rIdx < rowPerThread; ++rIdx)
        {
            int r = rIdx * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
            if (r >= nCoords)
                break;

            float sumVal(0);
            int clusterCount(0);
            for (int i = 0; i < nObjs; ++i)
            {
                if (membership[i] == c)
                {
                    sumVal += objects[r*nObjs + i];
                    clusterCount++;
                }
            }
            if (clusterCount > 0)
                clusters[nClusters*r+c] = sumVal / clusterCount;
        }
    }
}

__global__ static
void copy_rows(const float* src, const int sz1, const int sz2
                , const int copiedRows, float* dest
                , const int rowPerThread, const int colPerThread)
{
    for (int rIdx = 0; rIdx < rowPerThread; ++rIdx)
    {
        int r = rIdx * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
        if (r >= copiedRows)
            break;
            
        for (int cIdx = 0; cIdx < colPerThread; ++cIdx)
        {
            int c = cIdx * gridDim.y * blockDim.y + blockIdx.y * blockDim.y + threadIdx.y;
            if (c >= sz2)
                break;
            dest[c*copiedRows+r] = src[c*sz1+r];
        }
    }
}

int kMeans(float *deviceObjects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int     maxLoop,      /* maximum number of loops */
                   int    *membership,   /* out: [numObjs] */
                   float  *deviceClusters)
{
    int loop(0);
    float    delta;          /* % of objects change their clusters */
    int *deviceMembership;
    int *deviceIntermediates;

    CHECK_PARAM(deviceClusters, "deviceClusters cannot be NULL");
    
    //  To support reduction, numThreadsPerClusterBlock *must* be a power of
    //  two, and it *must* be no larger than the number of bits that will
    //  fit into an unsigned char, the type used to keep track of membership
    //  changes in the kernel.
    const unsigned int numThreadsPerClusterBlock = 128;
    const unsigned int numClusterBlocks =
        (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
#if BLOCK_SHARED_MEM_OPTIMIZATION
    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char) +
        numClusters * numCoords * sizeof(float);

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        err("WARNING: Your CUDA hardware has insufficient block shared memory. "
            "You need to recompile with BLOCK_SHARED_MEM_OPTIMIZATION=0. "
            "See the README for details.\n");
    }
#else
    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char);
#endif

    const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
    const unsigned int reductionBlockSharedDataSize = numReductionThreads * sizeof(unsigned int);

    CHECK_CUDA(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));

    // initialize membership[]
    if (membership)
    {
        for (int i=0; i<numObjs; i++) 
            membership[i] = -1;
        CHECK_CUDA(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));
    }
    else
    {
        int* hostMembership = (int*)malloc(numObjs*sizeof(int));
        CHECK_PARAM(hostMembership, "memory allocation failed");
        for (int i=0; i<numObjs; i++) 
            hostMembership[i] = -1;
        CHECK_CUDA(cudaMemcpy(deviceMembership, hostMembership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));
        free(hostMembership);
    }

    dim3 szGrid, szBlock;
    int rowPerThread, colPerThread;
        
    // initialize the cluster centroids
    get_kernel_config(numClusters, numCoords, szGrid, szBlock, rowPerThread, colPerThread);
    copy_rows<<<szGrid, szBlock>>>(deviceObjects, numObjs, numCoords
            , numClusters, deviceClusters, rowPerThread, colPerThread);
    
    do
    {
        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);

        //cudaDeviceSynchronize();
        //CHECK_CUDA(cudaGetLastError());

        compute_delta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>
            (deviceIntermediates, numClusterBlocks, numReductionThreads);

        //cudaDeviceSynchronize();
        //CHECK_CUDA(cudaGetLastError());

        get_kernel_config(numCoords, numClusters, szGrid, szBlock, rowPerThread, colPerThread);
        
        update_cluster <<< szGrid, szBlock >>> (deviceObjects, deviceMembership
                    , deviceClusters, numCoords, numObjs, numClusters, rowPerThread, colPerThread);
        
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());
        
        // inefficient memory transfer
        int d;
        CHECK_CUDA(cudaMemcpy(&d, deviceIntermediates,
                  sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d/numObjs;
    } 
    while (delta > threshold && loop++ < maxLoop);

    if (membership)
    {
        CHECK_CUDA(cudaMemcpy(membership, deviceMembership, 
              numObjs*sizeof(int), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaFree(deviceMembership));
    CHECK_CUDA(cudaFree(deviceIntermediates));

    return (loop + 1);
}

}