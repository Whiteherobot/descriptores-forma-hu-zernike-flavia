// ============================================================================
// Ruta:           cpp/apps/selftest.cpp
// Responsabilidad:
//     Prueba de validación del núcleo C++ SIN necesidad del Flavia Dataset.
//     Genera figuras sintéticas (círculo, cuadrado, triángulo, elipse) y verifica:
//       1) El descriptor de Fourier es INVARIANTE a rotación, traslación y escala.
//       2) La distancia Euclídea entre la misma forma (rotada/escalada) es
//          mucho menor que entre formas distintas -> el clasificador separa.
//     Además imprime el descriptor de un círculo para poder comparar el
//     resultado contra la referencia en Python (deben coincidir).
//
// Uso:  ./selftest
// ============================================================================
#include "shape_descriptor.h"

#include <cmath>
#include <iostream>
#include <vector>

// Dibuja una forma sintética sobre lienzo blanco (objeto oscuro).
static cv::Mat drawShape(const std::string& kind, int size, double scale,
                         double angleDeg, cv::Point shift) {
    cv::Mat img(size, size, CV_8UC1, cv::Scalar(255));
    cv::Point c(size / 2 + shift.x, size / 2 + shift.y);
    int r = static_cast<int>(size * 0.30 * scale);

    std::vector<cv::Point> poly;
    if (kind == "circle" || kind == "ellipse") {
        double ry = (kind == "ellipse") ? r * 0.55 : r;
        cv::ellipse(img, c, cv::Size(r, static_cast<int>(ry)), angleDeg,
                    0, 360, cv::Scalar(0), cv::FILLED);
        return img;
    }
    int n = (kind == "triangle") ? 3 : 4;
    double off = (kind == "square") ? CV_PI / 4 : -CV_PI / 2;
    for (int i = 0; i < n; ++i) {
        double a = off + angleDeg * CV_PI / 180.0 + 2 * CV_PI * i / n;
        poly.emplace_back(c.x + static_cast<int>(r * std::cos(a)),
                          c.y + static_cast<int>(r * std::sin(a)));
    }
    std::vector<std::vector<cv::Point>> polys{poly};
    cv::fillPoly(img, polys, cv::Scalar(0));
    return img;
}

int main() {
    std::cout << "=== SELF-TEST núcleo C++ (Parte 2) ===\n\n";

    // 1) Invarianza: mismo círculo con distinta rotación/escala/posición.
    auto base = vc::describeImage(drawShape("circle", 400, 1.0, 0, {0, 0}));
    auto rot  = vc::describeImage(drawShape("circle", 400, 1.0, 47, {0, 0}));
    auto scl  = vc::describeImage(drawShape("circle", 400, 1.6, 0, {0, 0}));
    auto mov  = vc::describeImage(drawShape("circle", 400, 1.0, 0, {40, -30}));

    std::cout << "Dim descriptor = " << base.size() << "\n";
    std::cout << "dist(circulo, mismo rotado 47°)  = "
              << vc::euclideanDistance(base, rot) << "\n";
    std::cout << "dist(circulo, mismo escala 1.6x) = "
              << vc::euclideanDistance(base, scl) << "\n";
    std::cout << "dist(circulo, mismo trasladado)  = "
              << vc::euclideanDistance(base, mov) << "\n";

    // 2) Separabilidad entre clases distintas.
    auto sq = vc::describeImage(drawShape("square", 400, 1.0, 0, {0, 0}));
    auto tr = vc::describeImage(drawShape("triangle", 400, 1.0, 0, {0, 0}));
    std::cout << "dist(circulo, cuadrado)          = "
              << vc::euclideanDistance(base, sq) << "\n";
    std::cout << "dist(circulo, triangulo)         = "
              << vc::euclideanDistance(base, tr) << "\n\n";

    // 3) Clasificación KNN sobre un mini-conjunto sintético.
    std::vector<vc::Sample> train = {
        {vc::describeImage(drawShape("circle",   400, 1.0,  10, {0, 0})), 0},
        {vc::describeImage(drawShape("circle",   400, 1.3,  80, {0, 0})), 0},
        {vc::describeImage(drawShape("square",   400, 1.0,  10, {0, 0})), 1},
        {vc::describeImage(drawShape("square",   400, 1.2,  35, {0, 0})), 1},
        {vc::describeImage(drawShape("triangle", 400, 1.0,  10, {0, 0})), 2},
        {vc::describeImage(drawShape("triangle", 400, 1.4,  60, {0, 0})), 2},
    };
    const char* names[] = {"circulo", "cuadrado", "triangulo"};
    int q0 = vc::classifyKNN(vc::describeImage(drawShape("circle",   400, 1.1, 200, {10, 5})), train, 3);
    int q1 = vc::classifyKNN(vc::describeImage(drawShape("square",   400, 0.9, 123, {0, 0})), train, 3);
    int q2 = vc::classifyKNN(vc::describeImage(drawShape("triangle", 400, 1.1, 250, {0, 0})), train, 3);
    std::cout << "KNN(circulo)   -> " << names[q0] << "\n";
    std::cout << "KNN(cuadrado)  -> " << names[q1] << "\n";
    std::cout << "KNN(triangulo) -> " << names[q2] << "\n\n";

    // 4) Imprimir descriptor del círculo (para comparar con Python).
    std::cout << "Descriptor circulo (primeros 8): ";
    for (int i = 0; i < 8 && i < (int)base.size(); ++i)
        std::cout << base[i] << " ";
    std::cout << "\n";

    bool invOK = vc::euclideanDistance(base, rot) < vc::euclideanDistance(base, sq);
    bool clsOK = (q0 == 0 && q1 == 1 && q2 == 2);
    std::cout << "\n[" << (invOK && clsOK ? "OK" : "FALLO")
              << "] invarianza=" << invOK << " clasificacion=" << clsOK << "\n";
    return (invOK && clsOK) ? 0 : 1;
}
