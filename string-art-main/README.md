# String-art
Implementazione C++, SIMD e CUDA di un algoritmo di String Art per generare immagini tramite linee rette.\
La String-Art è una tecnica di disegno che permette di ottenere immagini totalmente composte da linee rette. Per iniziare il procedimento, viene fornita in input un'immagine, sulla quale circonferenza vengono posizionati un numero di chiodi scelti dall'utente. A partire da questi chiodi, ne viene scelto uno casuale dal quale partiranno una serie di linee rette che lo collegheranno agli altri chiodi (esclusi quelli vicini). Tra tutte queste linee verrà scelta quella più scura, ovvero la linea che, una volta sovrapposta all'immagine originale, ha come somma dei valori dei pixel che la compongono il risultato più basso. In output si ottiene l'immagine creata attraverso questo algoritmo. 
Il problema è stato scelto in quanto permetteva un ampio margine di miglioramento nel tempo di esecuzione; questo è dovuto all'algoritmo che, per ogni potenziale linea, deve determinare quali pixel attraversa. Fare questa iterazione sulla CPU, in maniera sequenziale, richiede molto tempo. 

## Usage

Per runnare lo script

```
string_art input.pgm num_pins opacity threshold skipped_neighbors output.pgm
```

where
 - `input.pgm` is a square binary “P5” portable graymap image without comments (suggested `512x512`),
 - `num_pins` is the number of nails (suggested: `256`),
 - `opacity` is the opacity factor, see below (suggested: `0.2`, higher value means brighter image),
 - `threshold` is the opacity factor, see below (suggested: `255`, higher value means brighter image),
 - `skipped_neighbors` is how consecutive nails can be (suggested `32`),
 - `scale_factor` is the scaling factor of the output image (suggested: `8`),
 - `output` the output string-art image.

Esempio

```
string_art input.pgm 256 0.2 255 32 8 output.pgm
```

Per convertire immagine da pmg a png (e viceversa)
```
convert output.pgm -colorspace Gray output.png
```

## Risultati C++

![results](img/ada_s1.png)

## Risultati SIMD

![results](img/aec.png)

## Risultati Cuda Naive

![results](img/aec.png)

## Risultati CUDA

![results](img/aec.png)
