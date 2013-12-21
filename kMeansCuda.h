/* 
 * File:   kMeansCuda.h
 * Author: hvpham
 *
 * Created on December 22, 2013, 12:27 AM
 */

#ifndef KMEANSCUDA_H
#define	KMEANSCUDA_H

#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>

namespace cuda
{

inline void checkCudaError(cudaError_t err
                    , char const * file, unsigned int line)
{
    if (err != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA error " << err << " at " << file << ":" << line;
        throw std::runtime_error(ss.str());
    }
}

inline void check(bool bTrue, const char* msg
                     , char const * file, unsigned int line)
{
    if (!bTrue)
    {
        std::stringstream ss;
        ss << "Error: \"" << msg << "\" at " << file << ":" << line;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_PARAM(x, msg)   cuda::check((x), (msg), __FILE__, __LINE__)
#define CHECK_CUDA(cudaError) cuda::checkCudaError((cudaError), __FILE__, __LINE__)

// device memory, column-majored
int kMeans(float *deviceObjects,      /* in: [numObjs][numCoords] */
               int     numCoords,    /* no. features */
               int     numObjs,      /* no. objects */
               int     numClusters,  /* no. clusters */
               float   threshold,    /* % objects change membership */
               int     maxLoop,      /* maximum number of loops */
               int    *membership,   /* out: [numObjs] */
               float  *deviceClusters);

// original version: host memory, row-majored
float** kMeansHost(float **objects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   float   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
                   int    *loop_iterations);
}

#endif	/* KMEANSCUDA_H */

