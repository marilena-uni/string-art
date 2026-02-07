#include "StringArtist.h"
#include <immintrin.h>
#include <iostream>
#include "BresenhamLineIterator.h"
#include <chrono>
#include <cuda.h>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

/** * @brief check CUDA call for errors and exit on failure
 * @copyright 2025 Fabio Tosi, Alma Mater Studiorum - Università di Bologna
 */
#define CHECK(call)                                                                 \
{                                                                                   \
    const cudaError_t error = call;                                                 \
    if (error != cudaSuccess) {                                                     \
        fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n",                \
                __FILE__, __LINE__, error, cudaGetErrorString(error));              \
        std::exit(1);                                                               \
    }                                                                               \
}

namespace {
    float CANVAS_LINE_OPACITY = 1.0f;
}

cudaTextureObject_t texImage = 0;
cudaTextureObject_t texDraft = 0;

struct alignas(8) PinPos {
    int x, y;
};

__constant__ PinPos d_pins[4096];

struct ScoreResult {
    float score;
    int pinIndex;
};

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
}

__device__ void warpReduce(volatile float* shared_scores, volatile int* shared_pins, int tid) {
    if(shared_scores[tid] > shared_scores[tid + 32]) {
        shared_scores[tid] = shared_scores[tid + 32];
        shared_pins[tid] = shared_pins[tid + 32];
    }
    if(shared_scores[tid] > shared_scores[tid + 16]) {
        shared_scores[tid] = shared_scores[tid + 16];
        shared_pins[tid] = shared_pins[tid + 16];
    }
    if(shared_scores[tid] > shared_scores[tid + 8]) {
        shared_scores[tid] = shared_scores[tid + 8];
        shared_pins[tid] = shared_pins[tid + 8];
    }
    if(shared_scores[tid] > shared_scores[tid + 4]) {
        shared_scores[tid] = shared_scores[tid + 4];
        shared_pins[tid] = shared_pins[tid + 4];
    }
    if(shared_scores[tid] > shared_scores[tid + 2]) {
        shared_scores[tid] = shared_scores[tid + 2];
        shared_pins[tid] = shared_pins[tid + 2];
    }
    if(shared_scores[tid] > shared_scores[tid + 1]) {
        shared_scores[tid] = shared_scores[tid + 1];
        shared_pins[tid] = shared_pins[tid + 1];
    }
}

__global__ void drawLine_kernel(unsigned char* image, size_t pitch, int currentPinId, int nextPinId, const float opacity, int scale)
{
    int currentPin_x = d_pins[currentPinId].x * scale;
    int currentPin_y = d_pins[currentPinId].y * scale;
    int nextPin_x = d_pins[nextPinId].x * scale; 
    int nextPin_y = d_pins[nextPinId].y * scale;

    int diff_x = nextPin_x - currentPin_x;
    int diff_y = nextPin_y - currentPin_y;
    int distance = max(abs(diff_x), abs(diff_y));
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid <= distance){
        float t = (float)tid / (float)distance;
        int pixel_x = __float2int_rn(currentPin_x + t * diff_x);
        int pixel_y = __float2int_rn(currentPin_y + t * diff_y);

        unsigned char* row = image + (pixel_y * pitch);
        int value = (float)row[pixel_x] * (1.0f - opacity);
        row[pixel_x] = (unsigned char)value;
    }
}

__global__ void findNextPin_kernel(int currentPinId, cudaTextureObject_t texImg, 
    cudaTextureObject_t texDraft, float* d_scores, int m_numPins,
    bool* m_adjacency, int m_skippedNeighbors, size_t width)
{
    int tid = threadIdx.x;
    int nextPinId = blockIdx.x;

    if(nextPinId >= m_numPins) return;

    if (tid == 0) d_scores[nextPinId] = 1e30f;
    __syncthreads();

    int diff = abs(nextPinId - currentPinId);
    int dist = min(diff, m_numPins - diff);

    if (dist < m_skippedNeighbors || m_adjacency[currentPinId * m_numPins + nextPinId]) return;

    float currentPin_x = (float)d_pins[currentPinId].x;
    float currentPin_y = (float)d_pins[currentPinId].y;
    float nextPin_x = (float)d_pins[nextPinId].x; 
    float nextPin_y = (float)d_pins[nextPinId].y;

    float diff_x = nextPin_x - currentPin_x;
    float diff_y = nextPin_y - currentPin_y;
    int distance = (int)max(abs(diff_x), abs(diff_y));
    
    __shared__ float s_score;
    if (tid == 0) s_score = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i <= distance; i += blockDim.x) {
        float t = (float)i / (float)distance;
        int px = __float2int_rn(currentPin_x + t * diff_x);
        int py = __float2int_rn(currentPin_y + t * diff_y);
        
        float valImg = tex2D<float>(texImg, px + 0.5f, py + 0.5f); 
        float valDraft = tex2D<float>(texDraft, px + 0.5f, py + 0.5f);

        local_sum += (valImg * 255.0f) + (255.0f - (valDraft * 255.0f));
    }

    for (int offset = 16; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    if ((tid % 32) == 0) atomicAdd(&s_score, local_sum);
    __syncthreads();

    if (distance > 0 && tid == 0)
        d_scores[nextPinId] = s_score / (float)(distance + 1);
}

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

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) { 
        if (tid < s) {
            if(shared_scores[tid] > shared_scores[tid + s]) {
                shared_scores[tid] = shared_scores[tid + s];
                shared_pins[tid] = shared_pins[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce(shared_scores, shared_pins, tid);

    if(tid == 0) {
        d_finalResult->score = shared_scores[0];
        d_finalResult->pinIndex = shared_pins[0];
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
    auto start = std::chrono::high_resolution_clock::now();

    size_t w = m_imagePtr->size();
    size_t c = m_imagePtr->size() * m_scaleFactor;
    size_t canvas_size = c * c;

    unsigned char *d_canvas, *d_image_pitched, *d_draft_pitched;
    bool* d_adjacency;
    float *d_scores;
    size_t pitch_image, pitch_draft;

    CHECK(cudaMallocPitch(&d_image_pitched, &pitch_image, w, w));
    CHECK(cudaMallocPitch(&d_draft_pitched, &pitch_draft, w, w));
    CHECK(cudaMemcpy2D(d_image_pitched, pitch_image, m_imagePtr->getFirstPixelPointer(), w, w, w, cudaMemcpyHostToDevice));
    CHECK(cudaMemset2D(d_draft_pitched, pitch_draft, 255, w, w));

    cudaResourceDesc resDescImg, resDescDraft;
    memset(&resDescImg, 0, sizeof(resDescImg));
    resDescImg.resType = cudaResourceTypePitch2D;
    resDescImg.res.pitch2D.devPtr = d_image_pitched;
    resDescImg.res.pitch2D.width = w;
    resDescImg.res.pitch2D.height = w;
    resDescImg.res.pitch2D.pitchInBytes = pitch_image;
    resDescImg.res.pitch2D.desc = cudaCreateChannelDesc<unsigned char>();

    resDescDraft = resDescImg;
    resDescDraft.res.pitch2D.devPtr = d_draft_pitched;
    resDescDraft.res.pitch2D.pitchInBytes = pitch_draft;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;

    CHECK(cudaCreateTextureObject(&texImage, &resDescImg, &texDesc, NULL));
    CHECK(cudaCreateTextureObject(&texDraft, &resDescDraft, &texDesc, NULL));

    CHECK(cudaMalloc(&d_canvas, canvas_size));
    CHECK(cudaMalloc(&d_adjacency, m_numPins * m_numPins * sizeof(bool)));
    CHECK(cudaMalloc(&d_scores, m_numPins * sizeof(float)));
    CHECK(cudaMemset(d_adjacency, 0, m_numPins * m_numPins * sizeof(bool)));
    CHECK(cudaMemset(d_canvas, 255, canvas_size));

    std::vector<PinPos> h_pins(m_numPins);
    for(int i=0; i<m_numPins; ++i) {
        auto p = m_draft.getPin(i);
        h_pins[i] = { (int)p[0], (int)p[1] };
    }
    CHECK(cudaMemcpyToSymbol(d_pins, h_pins.data(), m_numPins * sizeof(PinPos)));

    ScoreResult *d_finalResult;
    ScoreResult h_finalResult;
    CHECK(cudaMalloc(&d_finalResult, sizeof(ScoreResult)));

    dim3 blockSize(256);

    while (true)
    {
        findNextPin_kernel<<<m_numPins, 128>>>(currentPinId, texImage, texDraft, d_scores, m_numPins, d_adjacency, m_skippedNeighbors, w);

        size_t smBytes_r = 1024 * (sizeof(float) + sizeof(int));
        bestResult_kernel<<<1, 1024, smBytes_r>>>(d_scores, m_numPins, d_finalResult);
        CHECK(cudaMemcpy(&h_finalResult, d_finalResult, sizeof(ScoreResult), cudaMemcpyDeviceToHost));

        if (h_finalResult.score >= m_threshold || h_finalResult.pinIndex == -1) break;

        int bestPin = h_finalResult.pinIndex;
        m_iteration++;

        int distance = max(abs(h_pins[bestPin].x - h_pins[currentPinId].x), abs(h_pins[bestPin].y - h_pins[currentPinId].y));

        dim3 gridSize_1((distance + blockSize.x - 1) / blockSize.x);
        drawLine_kernel<<<gridSize_1, blockSize>>>(d_draft_pitched, pitch_draft, currentPinId, bestPin, m_draftOpacity, 1);

        dim3 gridSize_2((distance * m_scaleFactor + blockSize.x - 1) / blockSize.x);
        drawLine_kernel<<<gridSize_2, blockSize>>>(d_canvas, c, currentPinId, bestPin, CANVAS_LINE_OPACITY, m_scaleFactor);

        updateAdjacency_kernel<<<1, 1>>>(d_adjacency, currentPinId, bestPin, m_numPins);
        currentPinId = bestPin;
    }

    CHECK(cudaMemcpy(const_cast<unsigned char*>(m_canvas.getFirstPixelPointer()), d_canvas, canvas_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy2D(const_cast<unsigned char*>(m_draft.getFirstPixelPointer()), w, d_draft_pitched, pitch_draft, w, w, cudaMemcpyDeviceToHost));

    cudaFree(d_adjacency); 
    cudaFree(d_scores); 
    cudaFree(d_finalResult); 
    cudaFree(d_canvas);
    cudaFree(d_image_pitched);
    cudaFree(d_draft_pitched);
    CHECK(cudaDestroyTextureObject(texImage));
    CHECK(cudaDestroyTextureObject(texDraft));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Done after " << m_iteration << " iterations. Tempo: " << diff.count() << "s" << std::endl;
}

void StringArtist::saveImage(std::FILE* outputFile)
{
    std::fprintf(outputFile, "P5\n%ld %ld\n255\n", m_canvas.size(), m_canvas.size());
    std::fwrite(m_canvas.getFirstPixelPointer(), m_canvas.size(), m_canvas.size(), outputFile);
    std::fclose(outputFile);
}