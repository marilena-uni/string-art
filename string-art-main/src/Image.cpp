#include "Image.h"

Image::Image(size_t imageSize) :
    m_imageSize(imageSize)
{
    m_data.resize(m_imageSize * m_imageSize, 255);
}

Image::Image(std::FILE* imageFile, size_t imageSize) :
    m_imageSize(imageSize)
{
    size_t totalPixels = m_imageSize * m_imageSize;
    m_data.resize(totalPixels);
    
    size_t readElements = std::fread(&m_data[0], 1, totalPixels, imageFile);
    
    if (readElements != totalPixels) {
        // Gestione errore: il file è più piccolo del previsto o corrotto
        // Per ora stampiamo un errore, garantendo la correttezza funzionale 
        std::fprintf(stderr, "Errore: letti solo %zu byte su %zu attesi.\n", readElements, totalPixels);
    }
    
    std::fclose(imageFile);
}

size_t Image::size() const
{
    return m_imageSize;
}

void Image::setPixelValue(const Point2D& pixel, unsigned char value)
{
    m_data[pixel[1] * m_imageSize + pixel[0]] = value;
}

unsigned char Image::getPixelValue(const Point2D& pixel) const
{
    return m_data[pixel[1] * m_imageSize + pixel[0]];
}

/*
unsigned char* Image::getFirstPixelPointer()
{
    return &m_data[0];
}*/

const unsigned char* Image::getFirstPixelPointer() const {
    return m_data.data(); // Se è un std::vector
    // oppure semplicemente: return m_data; (Se è già un puntatore)
}
