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

StringArtist::StringArtistcudaNaive(const Image& image, unsigned int numPins, float draftOpacity, float threshold, unsigned int skipped_neighbors, unsigned int scaleFactor) :
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

__global__ void findNextPin_kernel (int currentPinId, unsigned char* image, 
    unsigned char* d_draft, float* d_scores,int m_numPins , const PinPos* d_pins,
    bool* m_adjacency, int m_skippedNeighbors, size_t width)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("GPU: Kernel partito. currentPin: %d, numPins: %d\n", currentPinId, m_numPins);
        printf("pins %d %d \n", d_pins[0].x, d_pins[0].y);
         printf("pins %d %d \n", d_pins[1].x, d_pins[1].y);
          printf("pins %d %d \n", d_pins[2].x, d_pins[2].y);
    }
    
    int nextPinId = blockIdx.x * blockDim.x + threadIdx.x;

    if(nextPinId>=m_numPins) return; 
    d_scores[nextPinId] = 1e30f;

    int diff = abs(nextPinId - currentPinId);
    int dist = min(diff, m_numPins - diff);

    if (dist < m_skippedNeighbors || m_adjacency[currentPinId * m_numPins + nextPinId]) return;

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

    while(true){

        int pixel= currentPin_y * width + currentPin_x;
        score += (float) image[pixel] + (255 - d_draft[pixel]);
        ++pixelChanged;
        
        if (nextPin_x == currentPin_x && nextPin_y == currentPin_y) break;

        if (id==0 ) 
        {
           currentPin_x += incremento_x;
        } else {
          currentPin_y += incremento_y;
        }
        
        if (error > 0) {
            if (id==0 ) 
            {
               currentPin_y += incremento_y;
            } else {
              currentPin_x += incremento_x;
            }
            error -= 2 * deltaMain;             
        }
        error += 2 * deltaStep;
    }
   
    if (pixelChanged > 0)
    {
         d_scores[nextPinId]  = score/ (float) distance;
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
    size_t img_size = w * w;
    unsigned char *image, *d_draft;
    bool* d_adjacency ;
    float *d_scores;
    PinPos *d_pins;

    CHECK(cudaMalloc(&d_draft, img_size  ));
    CHECK(cudaMalloc(&image, img_size  ));
    CHECK(cudaMalloc(&d_adjacency, m_numPins * m_numPins * sizeof(bool)));
    CHECK(cudaMalloc(&d_scores, m_numPins * sizeof(float)));
    CHECK(cudaMalloc(&d_pins, m_numPins * sizeof(PinPos)));
    
    std::vector<PinPos> h_pins(m_numPins);
    for(int i=0; i<m_numPins; ++i) {
        auto p = m_draft.getPin(i);
        h_pins[i] = { (int)p[0], (int)p[1] };
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (m_numPins + threadsPerBlock - 1) / threadsPerBlock;

    CHECK(cudaMemcpy(d_pins, h_pins.data() , m_numPins * sizeof(PinPos), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(image, m_imagePtr->getFirstPixelPointer(), img_size  , cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_adjacency, 0 ,  m_numPins * m_numPins * sizeof(bool)));
    
    std::vector<float> h_scores(m_numPins);
    float bestScore = std::numeric_limits<float>::infinity();
    int bestPin = -1;
    
    while (true)
    {
        size_t nextPinId;
        bestPin = -1;
        bestScore = std::numeric_limits<float>::infinity();

        CHECK(cudaMemcpy(d_draft, m_draft.getFirstPixelPointer(), img_size, cudaMemcpyHostToDevice));

        findNextPin_kernel<<<blocksPerGrid, threadsPerBlock>>>
            (currentPinId, image, d_draft, d_scores, m_numPins ,  d_pins,
                  d_adjacency,  m_skippedNeighbors, w);

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
            return;
        }

        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_scores.data(), d_scores, m_numPins * sizeof(float), cudaMemcpyDeviceToHost));

        for(int i=0; i<m_numPins; i++)
        {
            if(h_scores[i] < bestScore)
            {
                bestScore=h_scores[i];
                bestPin= i;
            }
        }
        
      std::cout << "Iteration: " << m_iteration << " BestPin: " << bestPin << " Score: " << bestScore << " Threshold: " << m_threshold << std::endl;
       if (bestScore >= m_threshold || bestScore >= 1e29f || bestPin==-1) break;
        
        m_iteration++;

        bool val= true;
        drawLine(m_draft, currentPinId, bestPin, m_draftOpacity);
        drawLine(m_canvas, currentPinId, bestPin, CANVAS_LINE_OPACITY);

        CHECK(cudaMemcpy(&d_adjacency[currentPinId * m_numPins + bestPin], &val, sizeof(bool), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(&d_adjacency[bestPin * m_numPins + currentPinId], &val, sizeof(bool), cudaMemcpyHostToDevice));
        currentPinId = bestPin;
    }
  
    cudaFree(image); cudaFree(d_draft); cudaFree(d_adjacency); cudaFree(d_scores); cudaFree(d_pins);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Done after "<< m_iteration << " iterations" << std::endl;
    std::cout << "Tempo di esecuzione CPU: " << diff.count() << " secondi" << std::endl; 
}

void StringArtist::drawLine(StringArtImage& image, const size_t currentPinId, const size_t nextPinId, const float opacity)
{
    for (const Point2D& pixel : BresenhamLineIterator(image.getPin(currentPinId), image.getPin(nextPinId)))
    {
        int value = 0;
        if (opacity < 1.0f)
        {
            value = image.getPixelValue(pixel) * (1 - opacity);
        }
        image.setPixelValue(pixel, value);
    }
}

void StringArtist::saveImage(std::FILE* outputFile)
{
    std::fprintf(outputFile, "P5\n%ld %ld\n255\n", m_canvas.size(), m_canvas.size());
    std::fwrite(m_canvas.getFirstPixelPointer(), m_canvas.size(), m_canvas.size(), outputFile);
    std::fclose(outputFile);

}
