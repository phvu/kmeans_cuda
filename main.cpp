/* 
 * File:   main.cpp
 * Author: hvpham
 *
 * Created on December 22, 2013, 12:27 AM
 */

#include <cstdlib>
#include <cmath>
#include <assert.h>
#include <iostream>
#include <ctime>
#include "kMeansCuda.h"

float* createDataColMajored(int sz1, int sz2, bool cudaMalloc)
{
    // I use malloc() here just to make it coherent with createDataRowMajored()
    // you should use the C++ way...
    float* arr;
    if (cudaMalloc)
        CHECK_CUDA(cudaMallocHost(&arr, sz1*sz2*sizeof(float), cudaHostAllocDefault));
    else
        arr = (float*)malloc(sz1*sz2*sizeof(float));
    for (int i = 0; i < sz1; ++i)
        for (int j = 0; j < sz2; ++j)
        {
            arr[sz1*j + i] = i*100 + j;
        }
    return arr;
}

float** createDataRowMajored(int sz1, int sz2)
{
    float** ret = (float**)malloc(sz1*sizeof(float*));
    ret[0] = (float*)malloc(sz1*sz2*sizeof(float));
    for (int i = 1; i < sz1; ++i)
    {
        ret[i] = ret[i-1] + sz2;
    }
    for (int i = 0; i < sz1; ++i)
        for (int j = 0; j < sz2; ++j)
        {
            ret[i][j] = i*100 + j;
        }
    return ret;
}

float* callkMeans1(float* hostData, int nObjs, int nDim, int nClusters, int*& membership)
{
    float* devData, *devClusters, *hostClusters;
    CHECK_CUDA(cudaMalloc(&devData, nObjs*nDim*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(devData, hostData, nObjs*nDim*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&devClusters, nClusters*nDim*sizeof(float)));
    if (membership)
        membership = new int[nObjs];
    
    cuda::kMeans(devData, nDim, nObjs, nClusters, 0, 500, membership, devClusters);
    hostClusters = new float[nClusters*nDim*sizeof(float)];
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(hostClusters, devClusters, nClusters*nDim*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(devData));
    CHECK_CUDA(cudaFree(devClusters));
    
    return hostClusters;
}

float** callkMeans2(float** hostData, int nObjs, int nDim, int nClusters, int*& membership)
{
    int loops;
    membership = new int[nObjs];
    return cuda::kMeansHost(hostData, nDim, nObjs, nClusters, 0, membership, &loops);
}

void checkCorrectness()
{
    const int sz1 = 1024, sz2 = 1024, nClusters = 10;
    float* dataCm = createDataColMajored(sz1, sz2, false);
    float** dataRm = createDataRowMajored(sz1, sz2);
    int* membership1, *membership2;
    float *clusters1, **clusters2, *clusters3;
    
    clusters1 = callkMeans1(dataCm, sz1, sz2, nClusters, membership1);
    clusters2 = callkMeans2(dataRm, sz1, sz2, nClusters, membership2);
    
    for (int i = 0; i < sz1; ++i)
    {
        CHECK_PARAM(membership1[i] == membership2[i], "membership");
        //if(membership1[i] != membership2[i])
        //    std::cout << "Membership " << i << " " << membership1[i] << " " << membership2[i] << std::endl;
    }
    
    for (int i = 0; i < nClusters; ++i)
        for (int j = 0; j < sz2; ++j)
        {
            CHECK_PARAM(std::abs(clusters1[nClusters*j + i] - clusters2[i][j]) <= 1E-2, "clusters");
            //if(std::abs(clusters1[nClusters*j + i] - clusters2[i][j]) > 1E-2)
            //    std::cout << "Clusters " << i << " " << j << " " << clusters1[nClusters*j + i] 
            //                                  << " " << clusters2[i][j] 
            //                              << " " << clusters1[nClusters*j + i] - clusters2[i][j] << std::endl;
        }
    
    // membership = NULL is also fine
    int* dummyMembership = NULL;
    clusters3 = callkMeans1(dataCm, sz1, sz2, nClusters, dummyMembership);
    for (int i = 0; i < nClusters; ++i)
        for (int j = 0; j < sz2; ++j)
        {
            CHECK_PARAM(std::abs(clusters3[nClusters*j + i] - clusters2[i][j]) <= 1E-2, "clusters");
        }
    
    delete[] membership1;
    delete[] membership2;
    delete[] clusters1;
    delete[] clusters3;
    free(clusters2[0]);
    free(clusters2);
    free(dataCm);
    free(dataRm[0]);
    free(dataRm);
}

void benchMark()
{
    const int sz1 = 1024, sz2 = 1024, nClusters = 10;
    float* dataCm = createDataColMajored(sz1, sz2, true);
    float** dataRm = createDataRowMajored(sz1, sz2);
    int* membership1, *membership2;
    float *clusters1, **clusters2;
    const int TIMES = 100;
    
    {
        clock_t begin = clock();
        for (int i = 0; i < TIMES; ++i)
            clusters1 = callkMeans1(dataCm, sz1, sz2, nClusters, membership1);
        double elapsed_secs = double(clock() - begin) / CLOCKS_PER_SEC;
        std::cout << "callkMeans1: " << elapsed_secs << " secs" << std::endl;
    }
    
    {
        clock_t begin = clock();
        for (int i = 0; i < TIMES; ++i)
            clusters2 = callkMeans2(dataRm, sz1, sz2, nClusters, membership2);
        double elapsed_secs = double(clock() - begin) / CLOCKS_PER_SEC;
        std::cout << "callkMeans2: " << elapsed_secs << " secs" << std::endl;
    }
    
    delete[] membership1;
    delete[] membership2;
    delete[] clusters1;
    free(clusters2[0]);
    free(clusters2);
    CHECK_CUDA(cudaFreeHost(dataCm));
    free(dataRm[0]);
    free(dataRm);
}

int main(int argc, char** argv)
{
    checkCorrectness();
    benchMark();
    // callkMeans1: 116.61 secs
    // callkMeans2: 143.17 secs

    return 0;
}

