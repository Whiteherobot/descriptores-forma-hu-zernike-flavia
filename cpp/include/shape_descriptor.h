// ============================================================================
// Ruta:           cpp/include/shape_descriptor.h
// Responsabilidad:
//     Declarar el núcleo C++ NATIVO de la Parte 2: cálculo del Descriptor de
//     Fourier con la variante de COORDENADAS COMPLEJAS, segmentación del
//     contorno y clasificación por distancia Euclídea (vecino más cercano).
//
//     Este mismo archivo .cpp se compila tanto en la CLI de validación
//     (apps/main.cpp) como en la app Android (vía JNI), garantizando que el
//     algoritmo del teléfono y el de escritorio sean idénticos.
//
// Unidad: 3.2 (Descriptores de Fourier / Shape Signature).
// ============================================================================
#ifndef VC_SHAPE_DESCRIPTOR_H
#define VC_SHAPE_DESCRIPTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vc {

// Parámetros por defecto (coherentes con src/utils/config.py de la referencia
// en Python para que ambos produzcan el mismo vector).
constexpr int kResamplePoints = 128;     // CONTOUR_RESAMPLE_POINTS
constexpr int kDescriptorSize = 32;      // FOURIER_DESCRIPTOR_SIZE
constexpr int kKnnK = 7;                 // K = 7 (pista)
constexpr int kThreshold = 130;          // T = 130 (pista)
constexpr double TRAIN_RATIO = 0.80;     // 80% train / 20% test por clase

// Una muestra etiquetada: su descriptor de Fourier y su clase (14..20).
struct Sample {
    std::vector<float> descriptor;
    int label;
};

// --- Segmentación -----------------------------------------------------------
// Preprocesa una imagen (reduce ruido con filtro Gaussiano, binariza y extrae
// el contorno externo más grande con cv::findContours). Devuelve true y llena
// 'contour' si encuentra una figura.
bool extractLargestContour(const cv::Mat& imageBGRorGray,
                           std::vector<cv::Point>& contour,
                           bool useAdaptiveThreshold = false);

// --- Descriptor de Fourier (coordenadas complejas) --------------------------
// Remuestrea el contorno a N puntos equiespaciados por longitud de arco.
std::vector<cv::Point2f> resampleContour(const std::vector<cv::Point>& contour,
                                         int nPoints = kResamplePoints);

// Calcula el descriptor: s(n) = (x-xc)+j(y-yc) -> cv::dft -> |F(k)| / |F(1)|.
// Descarta F(0) (traslación), normaliza por |F(1)| (escala) y usa magnitudes
// (rotación). Devuelve un vector de 'nDescriptors' valores.
std::vector<float> fourierDescriptor(const std::vector<cv::Point>& contour,
                                     int nPoints = kResamplePoints,
                                     int nDescriptors = kDescriptorSize);

// Atajo: imagen -> contorno -> descriptor. Devuelve vector vacío si falla.
std::vector<float> describeImage(const cv::Mat& image,
                                 bool useAdaptiveThreshold = false);

// --- Clasificación ----------------------------------------------------------
// Distancia Euclídea entre dos descriptores (vista en clase con Hu).
float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b);

// Clasifica por K vecinos más cercanos (voto mayoritario) sobre el set de
// entrenamiento. Si k=1 equivale al vecino más cercano.
int classifyKNN(const std::vector<float>& query,
                const std::vector<Sample>& trainingSet,
                int k = kKnnK);

// --- Persistencia (modelo para el móvil) ------------------------------------
// Guarda/lee el set de entrenamiento como CSV: label,d0,d1,...,d{K-1}
bool saveModel(const std::string& path, const std::vector<Sample>& samples);
std::vector<Sample> loadModel(const std::string& path);

}  // namespace vc

#endif  // VC_SHAPE_DESCRIPTOR_H
