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

struct PinPos {
    int x, y;
};

__global__ void findNextPin_kernel (int currentPinId, unsigned char* image, 
    unsigned char* d_draft, float* d_scores,int m_numPins , const PinPos* d_pins,
    bool* m_adjacency, int m_skippedNeighbors, int width)
{
    
    int nextPinId = blockIdx.x * blockDim.x + threadIdx.x;
    d_scores[nextPinId] = 1e30f;

    int diff = nextPinId - currentPinId;
    int dist = min(diff % m_numPins , -diff % m_numPins);

    if (nextPinId>=m_numPins || dist < m_skippedNeighbors 
        || m_adjacency[currentPinId * m_numPins + nextPinId]) return;
    

    unsigned int pixelChanged = 0;
    float score = 0.0f;
    int currentPin_x = d_pins[currentPinId].x;
    int currentPin_y = d_pins[currentPinId].y;

    int nextPin_x = d_pins[nextPinId].x; ;
    int nextPin_y = d_pins[nextPinId].y;;

    int diff_x = nextPin_x - currentPin_x;
    int diff_y = nextPin_y - currentPin_y;

    int distance = max(abs(diff_x), abs(diff_y));
    int delta[2] = {abs(diff_x), abs(diff_y)};
    int id = abs(diff_x) >= abs(diff_y) ? 0 : 1;
    
    int increment[2];
    increment[id] = delta[id] >= 0 ? 1 : -1;
    increment[1 - id] = delta[1 - id] >= 0 ? 1 : -1;

    
    int error = 2 * delta[1 - id] - delta[id];
    int currentPin_xy[2] = {currentPin_x, currentPin_y};

    //posso anche usare delta perchè è la distanza e mi dice il num di pixel
    while (nextPin_x != currentPin_xy[0] && nextPin_y != currentPin_xy[1] ) {

        int pixel= currentPin_xy[1] * width + currentPin_xy[0];
        score += (float) image[pixel] + (255 - d_draft[pixel]);
        ++pixelChanged;
        
        currentPin_xy[id] += increment[id];

        if (error > 0)
        {   //  x = x + xi
            currentPin_xy[1 - id] += increment[1 - id];
            error -= 2 * delta[id]; //aggiorno errore
        }
        // per capire a pros pixel quanto mi sto allontanando dalla linea vera
        error += 2 * delta[1 - id]; 
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

    int w = static_cast<int>(std::sqrt(m_imagePtr->size()));
    size_t img_size = m_imagePtr->size();
    unsigned char *image, *d_draft;
    bool* d_adjacency ;
    float *d_scores;
    PinPos *d_pins;

    CHECK(cudaMalloc(&d_draft, img_size));
    CHECK(cudaMalloc(&image, img_size));
    CHECK(cudaMalloc(&d_adjacency, m_numPins * m_numPins * sizeof(bool)));
    CHECK(cudaMalloc(&d_scores, m_numPins * sizeof(float)));
    CHECK(cudaMalloc(&d_pins, m_numPins * sizeof(PinPos)));
    
    std::vector<PinPos> h_pins(m_numPins);
    for(int i=0; i<m_numPins; ++i) {
        auto p = m_draft.getPin(i);
        h_pins[i] = { (int)p[0], (int)p[1] };
    }

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((m_numPins + blockSize.x - 1) / blockSize.x,
                  (m_numPins + blockSize.y - 1) / blockSize.y);

    CHECK(cudaMemcpy(d_pins, h_pins.data() , m_numPins * sizeof(PinPos), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(image, m_imagePtr->getFirstPixelPointer() , img_size , cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_adjacency, 0 ,  m_numPins * m_numPins * sizeof(bool)));
    
    float* h_scores = (float*)malloc(sizeof(float) * m_numPins);
    float bestScore = std::numeric_limits<float>::infinity();
    int bestPin = -1;
    

    while (true)
    {
        size_t nextPinId;
        bestPin = -1;
        bestScore = std::numeric_limits<float>::infinity();

        CHECK(cudaMemcpy(d_draft, m_draft.getFirstPixelPointer() , m_imagePtr->size(), cudaMemcpyHostToDevice));

        findNextPin_kernel<<<gridSize, blockSize>>>
            (currentPinId, image, d_draft, d_scores, m_numPins ,  d_pins,
                  d_adjacency,  m_skippedNeighbors, w);

        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_scores, d_scores, m_numPins * sizeof(float), cudaMemcpyDeviceToHost));

        for(int i=0; i<m_numPins; i++)
        {
            if(h_scores[i] < bestScore)
            {
                bestScore=h_scores[i];
                bestPin= i;
            }
        }
        

        if (bestScore >= m_threshold || bestPin==-1) break;

        m_iteration++;
        bool* val;
        //std::cout << m_iteration << std::endl;
        drawLine(m_draft, currentPinId, bestPin, m_draftOpacity);
        drawLine(m_canvas, currentPinId, bestPin, CANVAS_LINE_OPACITY);

        CHECK(cudaMemcpy(&d_adjacency[currentPinId * m_numPins + bestPin], &val, sizeof(bool), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(&d_adjacency[bestPin * m_numPins + currentPinId], &val, sizeof(bool), cudaMemcpyHostToDevice));
        //m_adjacency[currentPinId][bestPin] = true;
        //m_adjacency[bestPin][currentPinId] = true;
        currentPinId = bestPin;

    }
    free(h_scores);
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
