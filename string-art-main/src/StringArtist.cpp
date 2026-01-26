
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


StringArtist::StringArtist(const Image& image, unsigned int numPins, float draftOpacity, float threshold, unsigned int skipped_neighbors, unsigned int scaleFactor) :
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
                //metto dentro la tabella con chiodi i e j  il linearIndex
                m_lineLUT[i][j].push_back(linearIndex);
                //push_back ->

                //per ogni linea io avrò anche il num di pixel
            }

            //siccome è una linea è uguale da una parte e dall'altra
            m_lineLUT[j][i] = m_lineLUT[i][j];
        }
    }

    //cosi ho tutte le linee che posso collegare a 0, quelle che possono collegare a 1 , eccc
    //e la tabella ha i pixel di ogni linea possibile già salvatiiiii
}



void StringArtist::windString()
{
    // Wind thread around pins until image can't be improved.
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
        //skippa i chiodi che sono troppo vicino a quello attuale 
        //skippa i chiodi dove ho già teso un filo
        int diff = static_cast<int>(nextPinId) - static_cast<int>(currentPinId);
        int dist = std::min(diff % m_numPins, -diff % m_numPins);
        if (dist < m_skippedNeighbors || m_adjacency[currentPinId][nextPinId])
            continue;

        unsigned int pixelChanged;

        //simula una linea tra il chiodo attuale e quello potenziale
        //carca la linea che passa sopra i pixel più scuri sull'imm origanlae che sono ancora scuri nel disegno
        float score = lineScore(currentPinId, nextPinId, pixelChanged);
       
        //se lo score è più basso , sta coprendo zone scure
        if (pixelChanged > 0 && score < bestScore)
        {
            bestScore = score;
            bestPinId = nextPinId;
        }
    }

    //restituisce true se ha trovaoto una linea abbastanza buona da essere tracciata
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
    //__m128i high_part = _mm_unpackhi_epi64(res, res); 
    //__m128i totalsum = _mm_add_epi64(res, high_part);

    //return (float)_mm_cvtsi128_si64(totalsum);

    /*
    float sum = 0.0f;

    //cosi da 8+8 doenta 16+16
    __m128i zero = _mm_setzero_si128();
    __m128i unpack_orig_l = _mm_unpacklo_epi8 (orig, zero);
    __m128i unpack_draft_l = _mm_unpacklo_epi8 (draft, zero);

    //cosi da 8+8 doenta 16+16
    //i miei dati quindi sono da 16 , quindi ho 8 celle da 16
    //invece di 16 celle da 8
    __m128i unpack_orig_hi = _mm_unpackhi_epi8 (orig, zero);
    __m128i unpack_draft_hi = _mm_unpackhi_epi8 (draft, zero);

    __m128i sum_first = _mm_add_epi16 (unpack_orig_l, unpack_draft_l);
    __m128i sum_last = _mm_add_epi16 (unpack_orig_hi, unpack_draft_hi);

    //__m128i final_v = _mm_add_epi16(sum_first, sum_last);
    __m128i sum0 = _mm_sad_epu8( sum_first, zero);
    __m128i sum1 = _mm_sad_epu8( sum_last, zero);

    __m128i sum2 = _mm_add_epi64(sum0, sum1);
    
    __m128i high_part = _mm_unpackhi_epi64(sum2, sum2); 
    __m128i totalsum = _mm_add_epi64(sum2, high_part);

    //ogni dato ora vale 16 perchè ho fatto l'unpack
    /*uint16_t first_b[8] _attribute_((aligned(16)));
    uint16_t last_b[8] _attribute_((aligned(16)));
   
    mm_store_si128 ((__m128i*)first_b, sum_first);
    _mm_store_si128 ((__m128i*)last_b, sum_last);
   
    for (int i=0; i<8; i++)
    {
        sum+=first_b[i];
        sum+=last_b[i];
    }
        //posso usare sad
    
    
    //hadd è leggermente meno conveniente di add o di sad perchè dee incrociare
   /* __m128i v_sum = _mm_add_epi16(sum_first, sum_last);

    v_sum = _mm_hadd_epi16(v_sum, v_sum); 
    
    v_sum = _mm_hadd_epi16(v_sum, v_sum); 

    v_sum = _mm_hadd_epi16(v_sum, v_sum); 

    return (float)_mm_extract_epi16(v_sum, 0);*/

    //return (float)_mm_extract_epi16(totalsum, 0); 
    // return (float)_mm_cvtsi128_si64(totalsum);
}
    
/*

float StringArtist::lineScore(const size_t currentPinId, const size_t nextPinId, unsigned int& pixelChanged) const
{
    pixelChanged = 0;
    float score = 0.f;
    Point2D currentPin = m_draft.getPin(currentPinId);
    Point2D nextPin = m_draft.getPin(nextPinId);
    Point2D diff = nextPin - currentPin;
    int distance = std::max(std::abs(diff[0]), std::abs(diff[1]));

        uint8_t pixel_strtt_draft[16] __attribute__((aligned(16)));
        uint8_t pixel_strtt_orig[16] __attribute__((aligned(16)));

        const uint8_t* raw_orig = m_imagePtr->getFirstPixelPointer();
        const uint8_t* raw_draft = m_draft.getFirstPixelPointer();
        __m128i v_255 = _mm_set1_epi8((unsigned char)255);
        int i=0; 
        size_t length = m_lineLUT[currentPinId][nextPinId].size();
        size_t k = 0;
        const std::vector<unsigned int>& currentLine = m_lineLUT[currentPinId][nextPinId];
        const unsigned int* lut_ptr = currentLine.data(); // Accesso diretto all'array della LUT

        while ( pixelChanged < length)
        {
            unsigned int idx = lut_ptr[i]; 
            pixel_strtt_orig[i % 16] = raw_orig[idx];    
            pixel_strtt_draft[i % 16] = raw_draft[idx];
            i++;
            ++pixelChanged;
                if (i == 16){
                __m128i orig = _mm_load_si128((const __m128i*) pixel_strtt_orig);
                __m128i draft_raw = _mm_load_si128((const __m128i*) pixel_strtt_draft);
                __m128i draft_inv = _mm_subs_epu8(v_255, draft_raw);
                score+=funzSIMD(orig, draft_inv);
                i=0;}
            
        }
       
       
         for (; k < length; ++k) {
        unsigned int idx = lut_ptr[k];
        score += (uint32_t)(raw_orig[idx] + (255 - raw_draft[idx]));
    }
       
        
    if ( i!=0 )
       {
         for (int j=i; j<16; j++)
         { 
            pixel_strtt_orig[j]=0;
            pixel_strtt_draft[j]=0;
         }
         __m128i orig = _mm_load_si128((const __m128i*)pixel_strtt_orig);
        __m128i draft = _mm_load_si128((const __m128i*)pixel_strtt_draft);
         score += funzSIMD(orig, draft);
       }


    return score / distance;
}

*/
/*
float StringArtist::lineScore(const size_t currentPinId, const size_t nextPinId, unsigned int& pixelChanged) const
{
    pixelChanged = 0;
    float score = 0.f;
    __m128i v_255 = _mm_set1_epi8((unsigned char)255);
    Point2D currentPin = m_draft.getPin(currentPinId);
    Point2D nextPin = m_draft.getPin(nextPinId);
    Point2D diff = nextPin - currentPin;

    //uso due buffer allineati
    uint8_t pixel_strtt_orig[16] __attribute__((aligned(16)));
    uint8_t pixel_strtt_draft[16] __attribute__((aligned(16)));

    //distanza tra i due chiodi
    int distance = std::max(std::abs(diff[0]), std::abs(diff[1]));
    int c, i = 0; 

    //calcolo quanti quadratini della griglia devono essere attraversati dal filo per andare da A a B
    for (const Point2D& pixel : BresenhamLineIterator(currentPin, nextPin))
    {
        //16 celle -> 16 pixel 
        pixel_strtt_orig[i] = m_imagePtr->getPixelValue(pixel);
        pixel_strtt_draft[i] = m_draft.getPixelValue(pixel);
        i++;
        ++pixelChanged;
        
        if (i==16)
        {
            __m128i orig = _mm_load_si128((const __m128i*)pixel_strtt_orig);
            __m128i draft_raw = _mm_load_si128((const __m128i*)pixel_strtt_draft);

            __m128i draft_inv = _mm_subs_epu8(v_255, draft_raw);

            score += funzSIMD(orig, draft_inv);
            i=0;
            //m_imagePtr->getPixelValue(pixel) + (255 - m_draft.getPixelValue(pixel));
        }
        //immagine orig 0-> nero, vogliamo uno score basso perche vogliamo rimanere sopra i neri
        //draft -> inverto il valore perchè se è tanto basso vicino allo 0 vuol dire che sono già passata tante volte ee quini lo voglio evitare
        //score basso = scurendo una zona che nell'origine è nera ma nel mio è ancora troppo chiara
    }

    if ( i!=0 )
       {
         for (int j=i; j<16; j++)
         { 
            pixel_strtt_orig[j]=0;
            pixel_strtt_draft[j]=0;
         }
         __m128i orig = _mm_load_si128((const __m128i*)pixel_strtt_orig);
        __m128i draft = _mm_load_si128((const __m128i*)pixel_strtt_draft);
         score += funzSIMD(orig, draft);
       }


    return score / distance;
}
*/


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

    /*size_t rem = length - k; // Quanti pixel mancano alla fine?
    if (rem > 0) {
        for (size_t j = 0; j < 16; j++) {
            if (j < rem) {
                unsigned int idx = indices[k + j];
                b_orig[j] = raw_orig[idx];
                b_draft[j] = raw_draft[idx];
            } else {
                // Riempio di zeri i posti vuoti
                b_orig[j] = 0;
                b_draft[j] = 255; // ATTENZIONE: metto 255 così (255-255) fa 0 nell'inversione
            }
        }
        
        __m128i v_orig = _mm_load_si128((__m128i*)b_orig);
        __m128i v_draft = _mm_load_si128((__m128i*)b_draft);
        __m128i v_draft_inv = _mm_subs_epu8(v_255, v_draft);
        
        score += funzSIMD(v_orig, v_draft_inv);
    }

*/
    // 3. Ritorna lo score diviso per la lunghezza reale
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

/*
#include "StringArtist.h"

#include <iostream>
#include "BresenhamLineIterator.h"
#include <chrono>

namespace {
    float CANVAS_LINE_OPACITY = 1.0f;
}

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

void StringArtist::windString()
{
    // Wind thread around pins until image can't be improved.
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
        
        if (pixelChanged > 0 && score < bestScore)
        {
            bestScore = score;
            bestPinId = nextPinId;
        }
    }
    return bestScore < m_threshold;
}

float StringArtist::lineScore(const size_t currentPinId, const size_t nextPinId, unsigned int& pixelChanged) const
{
    pixelChanged = 0;
    float score = 0.f;
    Point2D currentPin = m_draft.getPin(currentPinId);
    Point2D nextPin = m_draft.getPin(nextPinId);
    Point2D diff = nextPin - currentPin;
    int distance = std::max(std::abs(diff[0]), std::abs(diff[1]));

    for (const Point2D& pixel : BresenhamLineIterator(currentPin, nextPin))
    {
        score += m_imagePtr->getPixelValue(pixel) + (255 - m_draft.getPixelValue(pixel));
        ++pixelChanged;
    }
    return score / distance;
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
*/