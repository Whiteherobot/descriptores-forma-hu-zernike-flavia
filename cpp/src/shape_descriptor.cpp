// ============================================================================
// Ruta:           cpp/src/shape_descriptor.cpp
// Responsabilidad:
//     Implementación del núcleo nativo de la Parte 2. Replica EXACTAMENTE el
//     algoritmo de la referencia en Python
//     (src/descriptors/shape_signature/fourier_complex.py) para que la app
//     móvil y el análisis de escritorio produzcan el mismo descriptor.
//
// Flujo (guía Parte 2 / Unidad 3.2):
//     Segmentación (Gauss + umbral) -> findContours (externo) ->
//     remuestreo del contorno -> señal compleja s(n)=(x-xc)+j(y-yc) ->
//     cv::dft -> |F(k)| / |F(1)| (descarta F(0) y la fase) -> distancia Euclídea.
// ============================================================================
#include "shape_descriptor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>

namespace vc {

// ---------------------------------------------------------------------------
// Segmentación: imagen -> contorno externo más grande.
// ---------------------------------------------------------------------------
bool extractLargestContour(const cv::Mat& input,
                           std::vector<cv::Point>& contour,
                           bool useAdaptiveThreshold) {
    if (input.empty()) return false;

    cv::Mat gray;
    // 4 canales (RGBA/BGRA): viene de Android (Utils.bitmapToMat). 3 canales:
    // imagen BGR de disco (CLI). 1 canal: ya en escala de grises.
    if (input.channels() == 4) cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    else if (input.channels() == 3) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();

    // 1) Reducción de ruido con filtro Gaussiano (sugerido por la guía).
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // 2) Binarización. Para el dibujo en vivo se usa umbral ADAPTATIVO; para las
    //    hojas (fondo claro uniforme) Otsu segmenta el objeto oscuro.
    cv::Mat bin;
    if (useAdaptiveThreshold) {
        cv::adaptiveThreshold(blurred, bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY_INV, 31, 5);
    } else {
        cv::threshold(blurred, bin, 0, 255,
                      cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    }

    // 3) Morfología para cerrar pequeños huecos del borde.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);

    // 4) Contorno externo más grande.
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    if (contours.empty()) return false;

    contour = *std::max_element(
        contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    return contour.size() >= 5;
}

// ---------------------------------------------------------------------------
// Remuestreo del contorno a N puntos equiespaciados por longitud de arco.
// (Equivalente a resample_contour de la referencia en Python.)
// ---------------------------------------------------------------------------
std::vector<cv::Point2f> resampleContour(const std::vector<cv::Point>& contour,
                                         int nPoints) {
    std::vector<cv::Point2f> out(nPoints);
    const size_t n = contour.size();
    if (n == 0) return out;

    // Longitudes acumuladas del contorno cerrado.
    std::vector<double> cum(n + 1, 0.0);
    for (size_t i = 0; i < n; ++i) {
        cv::Point2f a = contour[i];
        cv::Point2f b = contour[(i + 1) % n];
        cum[i + 1] = cum[i] + cv::norm(b - a);
    }
    double total = cum[n];
    if (total <= 0) {
        for (int i = 0; i < nPoints; ++i) out[i] = contour[0];
        return out;
    }

    // Para cada objetivo de longitud, interpolar linealmente.
    for (int i = 0; i < nPoints; ++i) {
        double target = total * i / nPoints;   // endpoint=False
        // Búsqueda del segmento que contiene 'target'.
        size_t seg = std::upper_bound(cum.begin(), cum.end(), target) - cum.begin();
        if (seg == 0) seg = 1;
        if (seg > n) seg = n;
        double t = (target - cum[seg - 1]) / std::max(1e-12, cum[seg] - cum[seg - 1]);
        cv::Point2f a = contour[(seg - 1) % n];
        cv::Point2f b = contour[seg % n];
        out[i] = a + t * (b - a);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Descriptor de Fourier con coordenadas complejas usando cv::dft.
// ---------------------------------------------------------------------------
std::vector<float> fourierDescriptor(const std::vector<cv::Point>& contour,
                                     int nPoints, int nDescriptors) {
    std::vector<float> desc;
    if (contour.size() < 5) return desc;

    std::vector<cv::Point2f> pts = resampleContour(contour, nPoints);

    // Orientación consistente (antihoraria): cv::findContours puede devolver el
    // contorno en sentido horario, lo que coloca el armónico fundamental en
    // F(N-1) en vez de F(1). Forzamos área con signo positiva para que F(1) sea
    // SIEMPRE la frecuencia fundamental y la normalización por |F(1)| sea estable.
    double signedArea = 0.0;
    for (int i = 0; i < nPoints; ++i) {
        const cv::Point2f& a = pts[i];
        const cv::Point2f& b = pts[(i + 1) % nPoints];
        signedArea += a.x * b.y - b.x * a.y;
    }
    if (signedArea < 0) std::reverse(pts.begin(), pts.end());

    // Centroide.
    cv::Point2f c(0.f, 0.f);
    for (const auto& p : pts) c += p;
    c *= (1.0f / nPoints);

    // Señal compleja s(n) = (x-xc) + j (y-yc)  -> Mat N x 1, 2 canales.
    cv::Mat signal(nPoints, 1, CV_32FC2);
    for (int i = 0; i < nPoints; ++i)
        signal.at<cv::Vec2f>(i, 0) = cv::Vec2f(pts[i].x - c.x, pts[i].y - c.y);

    // DFT compleja (sugerencia textual de la guía).
    cv::Mat spectrum;
    cv::dft(signal, spectrum, cv::DFT_COMPLEX_OUTPUT);

    // Magnitudes |F(k)|.
    std::vector<float> mag(nPoints);
    for (int i = 0; i < nPoints; ++i) {
        cv::Vec2f v = spectrum.at<cv::Vec2f>(i, 0);
        mag[i] = std::sqrt(v[0] * v[0] + v[1] * v[1]);
    }

    // Normalización por |F(1)| (primer armónico) -> invarianza a escala.
    // Por robustez se toma el mayor de los dos armónicos fundamentales.
    float f1 = std::max(mag[1], mag[nPoints - 1]);
    if (f1 < 1e-12f) f1 = 1.0f;

    // Baja frecuencia: positivos F(1..half) y negativos F(N-half..N-1).
    int half = nDescriptors / 2;
    desc.reserve(nDescriptors);
    for (int k = 1; k <= half; ++k) desc.push_back(mag[k] / f1);
    for (int k = nPoints - half; k < nPoints; ++k) desc.push_back(mag[k] / f1);
    return desc;
}

std::vector<float> describeImage(const cv::Mat& image, bool useAdaptiveThreshold) {
    std::vector<cv::Point> contour;
    if (!extractLargestContour(image, contour, useAdaptiveThreshold)) return {};
    return fourierDescriptor(contour);
}

// ---------------------------------------------------------------------------
// Clasificación.
// ---------------------------------------------------------------------------
float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) { float d = a[i] - b[i]; s += d * d; }
    return std::sqrt(s);
}

int classifyKNN(const std::vector<float>& query,
                const std::vector<Sample>& trainingSet, int k) {
    if (trainingSet.empty() || query.empty()) return -1;

    // Distancias a todas las muestras de entrenamiento.
    std::vector<std::pair<float, int>> dists;
    dists.reserve(trainingSet.size());
    for (const auto& s : trainingSet)
        dists.push_back({euclideanDistance(query, s.descriptor), s.label});

    k = std::min<int>(k, dists.size());
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

    // Voto mayoritario entre los k vecinos más cercanos.
    std::map<int, int> votes;
    for (int i = 0; i < k; ++i) votes[dists[i].second]++;
    int best = -1, bestVotes = -1;
    for (const auto& kv : votes)
        if (kv.second > bestVotes) { bestVotes = kv.second; best = kv.first; }
    return best;
}

// ---------------------------------------------------------------------------
// Persistencia del modelo (CSV) para empaquetarlo como asset en el móvil.
// ---------------------------------------------------------------------------
bool saveModel(const std::string& path, const std::vector<Sample>& samples) {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    for (const auto& s : samples) {
        f << s.label;
        for (float v : s.descriptor) f << "," << v;
        f << "\n";
    }
    return true;
}

std::vector<Sample> loadModel(const std::string& path) {
    std::vector<Sample> samples;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ',');
        Sample s; s.label = std::stoi(tok);
        while (std::getline(ss, tok, ',')) s.descriptor.push_back(std::stof(tok));
        samples.push_back(std::move(s));
    }
    return samples;
}

}  // namespace vc
