
#pragma GCC target("ssse3")
#include "StringArtist.h"
#include <stdalign.h>
#include <iostream>
#include "BresenhamLineIterator.h"
#include <chrono>
#include <immintrin.h>
#include <smmintrin.h>

namespace {
    float CANVAS_LINE_OPACITY = 1.0f;
}

StringArtist::StringArtistSIMD(const Image& image, unsigned int numPins, float draftOpacity, float threshold, unsigned int skipped_neighbors, unsigned int scaleFactor) :
    m_imagePtr(&image),
    m_numPins(numPins),
    m_draftOpacity(draftOpacity),
    m_threshold(threshold),
    m_skippedNeighbors(skipped_neighbors),
    m_scaleFactor(scaleFactor),
    m_iteration(0)
{
    //m_numPins -> num tot di chiodi 
    m_canvas = StringArtImage(m_imagePtr->size() * m_scaleFactor, m_numPins);
    m_draft = StringArtImage(m_imagePtr->size(), m_numPins);
    m_adjacency.resize(m_imagePtr->size(), std::vector<bool>(m_imagePtr->size(), false));

    size_t imgSize = m_imagePtr->size();
    //dim LUT = numCHIODI * numCHIODI
    m_lineLUT.resize(m_numPins, std::vector<std::vector<unsigned int>>(m_numPins));

    std::cout << "Generazione LUT delle linee..." << std::endl;

    for (size_t i = 0; i < m_numPins; ++i) {
        for (size_t j = i + 1; j < m_numPins; ++j) {
            
            Point2D p1 = m_draft.getPin(i); //prendo un chiodo 
            Point2D p2 = m_draft.getPin(j); //partedno dal chiodo adiacente vado avanti

            //prendo i pixel tra questi 2 chiodi
            for (const Point2D& pixel : BresenhamLineIterator(p1, p2)) {
                //il pixel è rappresentato da linearIndex
                unsigned int linearIndex = pixel[1] * imgSize + pixel[0];
                m_lineLUT[i][j].push_back(linearIndex);

                //per ogni linea io avrò anche il num di pixel
            }

            //siccome è una linea è uguale da una parte e dall'altra
            m_lineLUT[j][i] = m_lineLUT[i][j];
        }
    }
}

void StringArtist::windString()
{
    size_t currentPinId = 0;
    std::cout << "start winding" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    while (true)
    {
        size_t nextPinId;
        if (!findNextPin(currentPinId, nextPinId))
            break;


        m_iteration++;
        //std::cout << m_iteration << std::endl;
        drawLine(m_draft, currentPinId, nextPinId, m_draftOpacity);
        drawLine(m_canvas, currentPinId, nextPinId, CANVAS_LINE_OPACITY);


        m_adjacency[currentPinId][nextPinId] = true;
        m_adjacency[nextPinId][currentPinId] = true;
        currentPinId = nextPinId;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Done after "<< m_iteration << " iterations" << std::endl;
    std::cout << "Tempo di esecuzione CPU: " << diff.count() << " secondi" << std::endl;
}

bool StringArtist::findNextPin(const size_t currentPinId, size_t& bestPinId) const
{
    float bestScore = std::numeric_limits<float>::infinity();


    for (size_t nextPinId = 0; nextPinId < m_numPins; ++nextPinId)
    {
        int diff = static_cast<int>(nextPinId) - static_cast<int>(currentPinId);
        int dist = std::min(diff % m_numPins, -diff % m_numPins);
        if (dist < m_skippedNeighbors || m_adjacency[currentPinId][nextPinId])
            continue;

        unsigned int pixelChanged;

        float score = lineScore(currentPinId, nextPinId, pixelChanged);
       
        //se lo score è più basso , sta coprendo zone scure
        if (pixelChanged > 0 && score < bestScore)
        {
            bestScore = score;
            bestPinId = nextPinId;
        }
    }

    //restituisco true se ha trovaoto una linea abbastanza buona da essere tracciata
    return bestScore < m_threshold;
}


static inline float funzSIMD(__m128i orig, __m128i draft)
{
    __m128i zero = _mm_setzero_si128();

    __m128i s0 = _mm_sad_epu8(orig, zero); // 2 numeri che sono la somma degli altri valori
    __m128i s1 = _mm_sad_epu8(draft, zero);

    __m128i res = _mm_add_epi64(s0, s1); //ho 2 elementi da 64 che sono la somma di s0 e s1

    uint64_t el_hi=_mm_extract_epi64 (res, 0);
    uint64_t el_lo=_mm_extract_epi64 (res, 1);

    return (float) (el_hi+el_lo);

}

//prima facevo un for con la funz B pixel per pixel
float StringArtist::lineScore(size_t p1, size_t p2, unsigned int& pixelChanged) const {
    const std::vector<unsigned int>& indices = m_lineLUT[p1][p2];
    size_t length = indices.size();
    pixelChanged=indices.size();
    
    // Accesso diretto ai dati grezzi
    const uint8_t* raw_orig = m_imagePtr->getFirstPixelPointer();
    const uint8_t* raw_draft = m_draft.getFirstPixelPointer();

    float score = 0.0f;
    __m128i v_255 = _mm_set1_epi8((unsigned char)255);
    
    alignas(16) uint8_t b_orig[16];
    alignas(16) uint8_t b_draft[16];

    size_t k = 0;
    for (; k + 15 < indices.size(); k += 16) {
        
        for (int m = 0; m < 16; ++m) {
            unsigned int idx = indices[k + m];
            b_orig[m] = raw_orig[idx];
            b_draft[m] = raw_draft[idx];
        }

        __m128i v_orig = _mm_load_si128((__m128i*)b_orig);
        __m128i v_draft = _mm_load_si128((__m128i*)b_draft);
        __m128i v_draft_inv = _mm_subs_epu8(v_255, v_draft);

        score += funzSIMD(v_orig, v_draft_inv);
    }
    
    if ((length-k) > 0)
       {
         for (int j=(length-k); j<16; j++)
         { 
            b_orig[j]=0;
            b_draft[j]=0;
         }
         __m128i orig = _mm_load_si128((const __m128i*)b_orig);
        __m128i draft = _mm_load_si128((const __m128i*)b_draft);
        __m128i v_draft_inv = _mm_subs_epu8(v_255, draft);

         score += funzSIMD(orig, v_draft_inv);
       }

    //Ritorna lo score diviso per la lunghezza reale
    return score / (float)length;
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