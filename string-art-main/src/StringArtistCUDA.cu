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

struct ScoreResult {
    float score;
    int pinIndex;
};

//unrolling manuale
__device__ void warpReduce(volatile float* shared_scores, volatile float* shared_pins, int tid) { // Volatile is important here
    
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
__global__ void drawLine_kernel(unsigned char* image, int currentPinId, int nextPinId, const float opacity, const PinPos* d_pins, size_t width)
{
    int currentPin_x = d_pins[currentPinId].x;
    int currentPin_y = d_pins[currentPinId].y;
    int nextPin_x = d_pins[nextPinId].x; 
    int nextPin_y = d_pins[nextPinId].y;

    int diff_x = nextPin_x - currentPin_x;
    int diff_y = nextPin_y - currentPin_y;

    int distance = max(abs(diff_x), abs(diff_y));
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid <= distance){
        float t=(float)tid / (float)distance;
        
        int pixel_x = __float2int_rn(currentPin_x + t* (nextPin_x-currentPin_x));
        int pixel_y = __float2int_rn(currentPin_y + t* (nextPin_y-currentPin_y));
    
        int pixel = pixel_y * width + pixel_x;
        int value = 0;
        if (opacity < 1.0f){
            value = image[pixel] * (1 - opacity);
        }
       image[pixel] = value;
    }
}


__global__ void findNextPin_kernel (int currentPinId, unsigned char* image, 
    unsigned char* d_draft, float* d_scores, int m_numPins , const PinPos* d_pins,  int* d_pins_fin,
    bool* m_adjacency, int m_skippedNeighbors, size_t width)
{
    extern __shared__ float shared_scores[];
    int* shared_pins = (int*)&shared_scores[blockDim.x];

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("GPU: Kernel partito. currentPin: %d, numPins: %d\n", currentPinId, m_numPins);
        printf("pins %d %d \n", d_pins[0].x, d_pins[0].y);
        //printf("pins %d %d \n", d_pins[1].x, d_pins[1].y);
        //printf("pins %d %d \n", d_pins[2].x, d_pins[2].y);
    }
    
    int tid = threadIdx.x;
    int nextPinId = blockIdx.x * blockDim.x + threadIdx.x;     //accesso coalescente/allineato

    int bestPin = -1;
    shared_scores[tid] = 1e30f;

    if(nextPinId<m_numPins) { 
    
        int diff = abs(nextPinId - currentPinId);
        int dist = min(diff, m_numPins - diff);
    
    
        if (!(dist < m_skippedNeighbors 
            || m_adjacency[currentPinId * m_numPins + nextPinId])) 
        {
    
        unsigned int pixelChanged = 0;
        float score = 0.0f;
        int currentPin_x = d_pins[currentPinId].x;
        int currentPin_y = d_pins[currentPinId].y;
    
        int nextPin_x = d_pins[nextPinId].x; 
        int nextPin_y = d_pins[nextPinId].y;
    
        int diff_x = nextPin_x - currentPin_x;
        int diff_y = nextPin_y - currentPin_y;
    
        int distance = max(abs(diff_x), abs(diff_y));
        int id = abs(diff_x) >= abs(diff_y) ? 0 : 1;
        
        int incremento_x = (diff_x>=0) ? 1 : -1;
        int incremento_y = (diff_y>=0) ? 1 : -1;
        
        int deltaMain = (id == 0) ? abs(diff_x) : abs(diff_y);
        int deltaStep = (id == 0) ? abs(diff_y) : abs(diff_x);
    
        int error = 2 * deltaStep - deltaMain;
        int x = currentPin_x, y = currentPin_y;
    
        //posso anche usare delta perchè è la distanza e mi dice il num di pixel
        //while (nextPin_x != currentPin_xy[0] && nextPin_y != currentPin_xy[1] ) {
        //for(int i=0; i<=distance; i++){
        while(true){
    
            int pixel= y * width + x;
            //printf("current x %d \t y %d \n", currentPin_x, currentPin_y);
            score += (float) image[pixel] + (255 - d_draft[pixel]);
            //printf("Score parziale: %f\n", score);
            ++pixelChanged;
            
            if (x == nextPin_x  && y == nextPin_y) break;
    
            if (id==0 ) 
            {
               x += incremento_x;
            } else {
              y += incremento_y;
            }
    
            
            if (error > 0) {
    
            if (id==0 ) 
            {
               y += incremento_y;
            } else {
              x += incremento_x;
            }
                //currentPin_xy[1 - id] += deltaStep; 
                error -= 2 * deltaMain;             
            }
            error += 2 * deltaStep;
          }
        }
        
        if (pixelChanged > 0)
        {
             shared_scores[tid]  = score/ (float) distance;
             shared_pins[tid] = nextPinId;
        }
        }
        else {
            shared_scores[tid] = 1e30f;
            shared_pins[tid] = -1;
        }

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
    
        //il bestScore finisce in shared_scores[0]
        if(tid==0){
            d_scores[blockIdx.x] = shared_scores[0];
            d_pins_fin[blockIdx.x] = shared_pins[0];
        }
}

//riduzione parallela
__global__ void bestResult_kernel(const float *d_scores, const int *d_pins_fin, int gS, ScoreResult* d_finalResult)
{
    extern __shared__ float shared_scores[];
    int* shared_pins = (int*)&shared_scores[blockDim.x];
    
    int tid = threadIdx.x;

    if (tid < gS) {
        shared_scores[tid] = d_scores[tid];
        shared_pins[tid]   = d_pins_fin[tid];
    } else {
        shared_scores[tid] = 1e30f;
        shared_pins[tid]   = -1;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) { 
    if (tid < s)
    {
        if(shared_scores[tid] > shared_scores[tid + s])
        {
            shared_scores[tid] = shared_scores[tid + s];
            shared_pins[tid] = shared_pins[tid+s];
        }
    }
        
    __syncthreads();}

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
    // Wind thread around pins until image can't be improved.
    size_t currentPinId = 0;
    size_t device_currentPinId = 0;
    std::cout << "start winding" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    size_t w = m_imagePtr->size();
    std::cout << "img size" << w << std::endl;
    size_t img_size = w * w;
    unsigned char *image, *d_draft, *d_canvas;
    bool* d_adjacency ;
    float *d_scores;
    PinPos *d_pins;
    int *d_pins_fin;

    CHECK(cudaMalloc(&d_draft, img_size  ));
    CHECK(cudaMalloc(&d_canvas, img_size  ));
    CHECK(cudaMalloc(&image, img_size  ));
    CHECK(cudaMalloc(&d_adjacency, m_numPins * m_numPins * sizeof(bool)));
    CHECK(cudaMalloc(&d_scores, m_numPins * sizeof(float)));
    CHECK(cudaMalloc(&d_pins, m_numPins * sizeof(PinPos)));
    CHECK(cudaMalloc(&d_pins_fin, m_numPins * sizeof(int)));
    
    std::vector<PinPos> h_pins(m_numPins);


    for(int i=0; i<m_numPins; ++i) {
        auto p = m_draft.getPin(i);
        h_pins[i] = { (int)p[0], (int)p[1] };
    }

    //1D
    dim3 blockSize(256); 
    dim3 gridSize((m_numPins + blockSize.x - 1) / blockSize.x);  // Ceiling division

    ScoreResult *d_finalResult;
    ScoreResult h_finalResult;
    CHECK(cudaMalloc(&d_finalResult, sizeof(ScoreResult)));

    CHECK(cudaMemcpy(d_pins, h_pins.data() , m_numPins * sizeof(PinPos), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(image, m_imagePtr->getFirstPixelPointer(), img_size  , cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_adjacency, 0 ,  m_numPins * m_numPins * sizeof(bool)));
    
    std::vector<float> h_scores(m_numPins);
    float bestScore = std::numeric_limits<float>::infinity();
    int bestPin = -1;
    CHECK(cudaMemset(d_draft, 255, img_size));
    CHECK(cudaMemset(d_canvas, 255, img_size));

    while (true)
    {
        
        size_t nextPinId;
        bestPin = -1;
        bestScore = std::numeric_limits<float>::infinity();

       size_t smBytes = blockSize.x * (sizeof(float) + sizeof(int));

        findNextPin_kernel<<<gridSize, blockSize , smBytes>>>
            (currentPinId, image, d_draft, d_scores, m_numPins ,  d_pins, d_pins_fin,
                  d_adjacency,  m_skippedNeighbors, w);

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
            return;
        }

        int numThreads = 1024;

        size_t smBytes_r = numThreads * (sizeof(float) + sizeof(int));

        bestResult_kernel<<<1, numThreads, smBytes_r>>>(d_scores, d_pins_fin, gridSize.x, d_finalResult);
        
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(&h_finalResult, d_finalResult, sizeof(ScoreResult), cudaMemcpyDeviceToHost));

      bestPin = h_finalResult.pinIndex; bestScore =  h_finalResult.score;
       std::cout << "Iteration: " << m_iteration << " BestPin: " << bestPin << " Score: " << bestScore << " Threshold: " << m_threshold << std::endl;
      if (bestScore >= m_threshold || bestScore >= 1e29f || bestPin==-1) break;
        
        m_iteration++;
        bool val= true;

        int currentPin_x = h_pins[currentPinId].x;
        int currentPin_y = h_pins[currentPinId].y;
        int nextPin_x = h_pins[bestPin].x; 
        int nextPin_y = h_pins[bestPin].y;

        int diff_x = nextPin_x - currentPin_x;
        int diff_y = nextPin_y - currentPin_y;

        int distance = max(abs(diff_x), abs(diff_y));

        dim3 gridSize_1((distance + blockSize.x ) / blockSize.x);  // Ceiling division

        drawLine_kernel<<<gridSize_1, blockSize>>>
        ( d_draft,  currentPinId,  bestPin,  m_draftOpacity, d_pins, w);

        drawLine_kernel<<<gridSize_1, blockSize>>>
        ( d_canvas,  currentPinId,  bestPin,  CANVAS_LINE_OPACITY, d_pins, w);

        //invece di aggiornare d_adjacency con cudaMemcopy che prende tanto tempo preferisco creare un piccolo kernel
        updateAdjacency_kernel<<<1, 1>>>(d_adjacency, currentPinId, bestPin, m_numPins);
       
        currentPinId = bestPin;
    }

    CHECK(cudaMemcpy(m_canvas.getFirstPixelPointer(), d_canvas, img_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_draft.getFirstPixelPointer(), d_draft, img_size, cudaMemcpyDeviceToHost));

    cudaFree(image); cudaFree(d_draft); cudaFree(d_adjacency); cudaFree(d_scores); cudaFree(d_pins);
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
