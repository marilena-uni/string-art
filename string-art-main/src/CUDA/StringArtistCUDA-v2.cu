#include "StringArtist.h"
#include <immintrin.h>
#include <iostream>
#include "BresenhamLineIterator.h"
#include <chrono>
#include <cuda.h>
#include <cmath>
#include <cuda_runtime.h>
#include <driver_functions.h>
#define BLOCK_SIZE 16

/** 
 * @brief check CUDA call for errors and exit on failure
 * @param call the CUDA call to check
 * 
 * @copyright 2025 Fabio Tosi, Alma Mater Studiorum - Università di Bologna
 * 
 */

#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n",        \
                __FILE__, __LINE__, error, cudaGetErrorString(error));      \
        std::exit(1);                                                       \
    }                                                                       \
}

namespace {
    float CANVAS_LINE_OPACITY = 1.0f;
}

//Definire manualmente prima la dimensione del blocco (numero di thread per blocco).
//Poi, calcolare automaticamente la dimensione della griglia in base ai dati e alla dimensione del blocco
StringArtist::StringArtist(const Image& image, unsigned int numPins, float draftOpacity, float threshold, unsigned int skipped_neighbors, unsigned int scaleFactor) :
    m_imagePtr(&image),
    m_numPins(numPins),
    m_draftOpacity(draftOpacity),
    m_threshold(threshold),
    m_skippedNeighbors(skipped_neighbors),
    m_scaleFactor(scaleFactor),
    m_iteration(0)
{
    m_canvas = StringArtImage(m_imagePtr->size() * m_scaleFactor, m_numPins);
    m_draft = StringArtImage(m_imagePtr->size(), m_numPins);
    m_adjacency.resize(m_imagePtr->size(), std::vector<bool>(m_imagePtr->size(), false));
}

struct alignas(8) PinPos {
    int x, y;
};
__constant__ PinPos d_pins[4096];


struct ScoreResult {
    float score;
    int pinIndex;
};

//unrolling manuale
__device__ void warpReduce(volatile float* shared_scores, volatile int* shared_pins, int tid) { // Volatile is important here
    
    //sto confrontando thread fino al tid che è min di 32
    if(shared_scores[tid] > shared_scores[tid + 32])
        {
            shared_scores[tid] = shared_scores[tid + 32];
            shared_pins[tid] = shared_pins[tid+32];
        }
    if(shared_scores[tid] > shared_scores[tid + 16])
        {
            shared_scores[tid] = shared_scores[tid + 16];
            shared_pins[tid] = shared_pins[tid+16];
        }
    if(shared_scores[tid] > shared_scores[tid + 8])
        {
            shared_scores[tid] = shared_scores[tid + 8];
            shared_pins[tid] = shared_pins[tid+8];
        }
    if(shared_scores[tid] > shared_scores[tid + 4])
        {
            shared_scores[tid] = shared_scores[tid + 4];
            shared_pins[tid] = shared_pins[tid+4];
        }
    if(shared_scores[tid] > shared_scores[tid + 2])
        {
            shared_scores[tid] = shared_scores[tid + 2];
            shared_pins[tid] = shared_pins[tid+2];
        }
    if(shared_scores[tid] > shared_scores[tid + 1])
        {
            shared_scores[tid] = shared_scores[tid + 1];
            shared_pins[tid] = shared_pins[tid+1];
        }
}

//1 thread = 1 pixel
__global__ void drawLine_kernel(unsigned char* image, int currentPinId, int nextPinId, const float opacity, size_t width, int scale)
{
    int currentPin_x = d_pins[currentPinId].x *scale;
    int currentPin_y = d_pins[currentPinId].y * scale;
    int nextPin_x = d_pins[nextPinId].x * scale; 
    int nextPin_y = d_pins[nextPinId].y * scale;

    int diff_x = nextPin_x - currentPin_x;
    int diff_y = nextPin_y - currentPin_y;

    int distance = max(abs(diff_x), abs(diff_y));
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid <= distance){
    float t=(float)tid / (float)distance;
    //int pixel_x = currentPin_x + t* (nextPin_x-currentPin_x) ; 
    //int pixel_y = currentPin_y + t* (nextPin_y-currentPin_y) ; 
    
    //Round Nearest
    int pixel_x = __float2int_rn(currentPin_x + t* (nextPin_x-currentPin_x));
    int pixel_y = __float2int_rn(currentPin_y + t* (nextPin_y-currentPin_y));

    int pixel = pixel_y * width + pixel_x;
    int value = 0;
    if (opacity < 1.0f)
    {
        value = image[pixel] * (1 - opacity);
    }

    image[pixel] = value;
    }
}


__global__ void findNextPin_kernel (int currentPinId, unsigned char* image, 
    unsigned char* d_draft, float* d_scores, int m_numPins ,
    bool* m_adjacency, int m_skippedNeighbors, size_t width)
{
    
    int tid = threadIdx.x;
    
    int nextPinId = blockIdx.x;

      if(nextPinId>=m_numPins) return;

    int bestPin = -1;
    
    if (tid == 0) d_scores[nextPinId] = 1e30f;
    __syncthreads();


    int diff = abs(nextPinId - currentPinId);
    int dist = min(diff, m_numPins - diff);


    if ((dist < m_skippedNeighbors 
        || m_adjacency[currentPinId * m_numPins + nextPinId]))  return;


    unsigned int pixelChanged = 0;
    
    float currentPin_x = (float)d_pins[currentPinId].x;
    float currentPin_y = (float)d_pins[currentPinId].y;

    float nextPin_x = (float)d_pins[nextPinId].x; 
    float nextPin_y =(float) d_pins[nextPinId].y;

    float diff_x = nextPin_x - currentPin_x;
    float diff_y = nextPin_y - currentPin_y;
    float score = 0.0f;
    int distance =(int) max(abs(diff_x), abs(diff_y));
    
    //se no avrei 128 threads che contemporanema provano a mettere s_score=0
     __shared__ float s_score;
    if (tid == 0) s_score = 0.0f;
    __syncthreads();

    //posso anche usare delta perchè è la distanza e mi dice il num di pixel
    //while (nextPin_x != currentPin_xy[0] && nextPin_y != currentPin_xy[1] ) {
    //for(int i=0; i<=distance; i++){
    //while(true){
    float local_sum = 0.0f;
    for (int i = tid ; i <= distance ; i+= blockDim.x) {
        //interpo lineare
        float t = (float)i / (float)distance;
        int px = __float2int_rn(currentPin_x + t * diff_x);
        int py = __float2int_rn(currentPin_y + t * diff_y);
        int pixel = py * width + px;
        local_sum+=(float)__ldg(&image[pixel]) + (255.0f - (float)__ldg(&d_draft[pixel]));
    }

    atomicAdd(&s_score, local_sum);
    __syncthreads();

   
    if (distance > 0 && tid==0)
    {
         d_scores[nextPinId]  = s_score/ (float) (distance+1);
    }

}

//riduzione parallela
__global__ void bestResult_kernel(const float *d_scores, int gS, ScoreResult* d_finalResult)
{
    extern __shared__ float shared_scores[];
    int* shared_pins = (int*)&shared_scores[blockDim.x];
    
    int tid = threadIdx.x;

    float best_s = 1e30f;
    int best_p = -1;

    for (int i = tid; i < gS; i += blockDim.x) {
        float val = d_scores[i];
        if (val < best_s) {
            best_s = val;
            best_p = i;
        }
    }

    shared_scores[tid] = best_s;
    shared_pins[tid] = best_p;

    __syncthreads();

    #pragma unroll  //elimino così i brach
    for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) { 
    if (tid < s)
    {
        if(shared_scores[tid] > shared_scores[tid + s])
        {
            shared_scores[tid] = shared_scores[tid + s];
            shared_pins[tid] = shared_pins[tid+s];
        }
    }
        
    __syncthreads();}

    if (tid <32) warpReduce(shared_scores, shared_pins,tid);

    if(tid==0){
    d_finalResult->score = shared_scores[0];
    d_finalResult->pinIndex= shared_pins[0];
    }
     
}

__global__ void updateAdjacency_kernel(bool* d_adjacency, int pinA, int pinB, int numPins) {
    if (threadIdx.x == 0) {
        d_adjacency[pinA * numPins + pinB] = true;
        d_adjacency[pinB * numPins + pinA] = true;
    }
}

void StringArtist::windString()
{
    int currentPinId = 0;
    size_t device_currentPinId = 0;
    std::cout << "start winding" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    size_t w = m_imagePtr->size();
    size_t c = m_imagePtr->size() * m_scaleFactor;
    std::cout << "img size" << w << std::endl;
    size_t img_size = w * w;
    size_t canvas_size=  c * c;
    unsigned char *image, *d_draft, *d_canvas;
    bool* d_adjacency ;
    float *d_scores;
    //PinPos *d_pins;
    int *d_pins_fin;
    
    
    CHECK(cudaMalloc(&d_draft, img_size  ));
    CHECK(cudaMalloc(&d_canvas, canvas_size  ));
    CHECK(cudaMalloc(&image, img_size  ));
    CHECK(cudaMalloc(&d_adjacency, m_numPins * m_numPins * sizeof(bool)));
    CHECK(cudaMalloc(&d_scores, m_numPins * sizeof(float)));
    CHECK(cudaMalloc(&d_pins_fin, m_numPins * sizeof(int)));
    
    std::vector<PinPos> h_pins(m_numPins);

    for(int i=0; i<m_numPins; ++i) {
        auto p = m_draft.getPin(i);
        h_pins[i] = { (int)p[0], (int)p[1] };
    }

    //1D
    dim3 blockSize(256); 
    dim3 gridSize((m_numPins + blockSize.x - 1) / blockSize.x); 

    ScoreResult *d_finalResult;
    ScoreResult h_finalResult;
    CHECK(cudaMalloc(&d_finalResult, sizeof(ScoreResult)));

    CHECK(cudaMemcpyToSymbol(d_pins, h_pins.data() , m_numPins * sizeof(PinPos)));
    CHECK(cudaMemcpy(image, m_imagePtr->getFirstPixelPointer(), img_size  , cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_adjacency, 0 ,  m_numPins * m_numPins * sizeof(bool)));
    
    std::vector<float> h_scores(m_numPins);
    float bestScore = std::numeric_limits<float>::infinity();
    int bestPin = -1;
    CHECK(cudaMemset(d_draft, 255, img_size));
    CHECK(cudaMemset(d_canvas, 255, canvas_size));

    while (true)
    {
        
        size_t nextPinId;
        bestPin = -1;
        bestScore = std::numeric_limits<float>::infinity();

         findNextPin_kernel<<<m_numPins, 128>>>
            (currentPinId, image, d_draft, d_scores, m_numPins,
                  d_adjacency,  m_skippedNeighbors, w);

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
            return;
        }

        size_t smBytes_r = 1024 * (sizeof(float) + sizeof(int));

        bestResult_kernel<<<1, 1024, smBytes_r>>>(d_scores, m_numPins, d_finalResult);

        CHECK(cudaMemcpy(&h_finalResult, d_finalResult, sizeof(ScoreResult), cudaMemcpyDeviceToHost));

      bestPin = h_finalResult.pinIndex; bestScore =  h_finalResult.score;
       std::cout << "Iteration: " << m_iteration << " BestPin: " << bestPin << " Score: " << bestScore << " Threshold: " << m_threshold << std::endl;
      if (bestScore >= m_threshold || bestScore >= 1e29f || bestPin==-1) break;
        
        m_iteration++;
        //std::cout << "Num "<< m_iteration  << std::endl ;

        bool val= true;
        //std::cout << m_iteration << std::endl;

        //num threads = line lenght 
        int currentPin_x = h_pins[currentPinId].x;
        int currentPin_y = h_pins[currentPinId].y;
        int nextPin_x = h_pins[bestPin].x; 
        int nextPin_y = h_pins[bestPin].y;

        int diff_x = nextPin_x - currentPin_x;
        int diff_y = nextPin_y - currentPin_y;

        int distance = max(abs(diff_x), abs(diff_y));

        dim3 gridSize_1((distance + blockSize.x ) / blockSize.x);  // Ceiling division

        size_t draft_w = m_draft.size();   
        size_t canvas_w = m_canvas.size();

        drawLine_kernel<<<gridSize_1, blockSize>>>
        ( d_draft,  currentPinId,  bestPin,  m_draftOpacity, draft_w, 1);

        dim3 gridSize_2((distance*m_scaleFactor + blockSize.x ) / blockSize.x);  // Ceiling division
        drawLine_kernel<<<gridSize_2, blockSize>>>
        ( d_canvas,  currentPinId,  bestPin,  CANVAS_LINE_OPACITY, canvas_w, m_scaleFactor);
     
        updateAdjacency_kernel<<<1, 1>>>(d_adjacency, currentPinId, bestPin, m_numPins);
        currentPinId = bestPin;

    }

    CHECK(cudaMemcpy(const_cast<unsigned char*>(m_canvas.getFirstPixelPointer()), d_canvas, canvas_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(const_cast<unsigned char*>(m_draft.getFirstPixelPointer()), d_draft, img_size, cudaMemcpyDeviceToHost));

    cudaFree(image); cudaFree(d_draft); cudaFree(d_adjacency); cudaFree(d_scores); 
    cudaFree(d_pins_fin);  cudaFree(d_finalResult); cudaFree(d_canvas);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Done after "<< m_iteration << " iterations" << std::endl;
    std::cout << "Tempo di esecuzione CPU: " << diff.count() << " secondi" << std::endl; 
}

void StringArtist::saveImage(std::FILE* outputFile)
{
    std::fprintf(outputFile, "P5\n%ld %ld\n255\n", m_canvas.size(), m_canvas.size());
    std::fwrite(m_canvas.getFirstPixelPointer(), m_canvas.size(), m_canvas.size(), outputFile);
    std::fclose(outputFile);
}