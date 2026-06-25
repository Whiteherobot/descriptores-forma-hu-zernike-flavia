// ============================================================================
// Ruta:           cpp/apps/main.cpp
// Responsabilidad:
//     CLI de validación de escritorio para la PARTE 2. Entrena y evalúa el
//     reconocimiento de hojas (Flavia Dataset) con el MISMO núcleo C++ que usa
//     la app Android:
//         - Lee all.csv (count,id,y) y filtra las clases asignadas (14..20).
//         - Calcula el descriptor de Fourier (coordenadas complejas) de cada hoja.
//         - Divide 80% train / 20% test por clase.
//         - Clasifica por distancia Euclídea (KNN, K=7).
//         - Imprime accuracy y matriz de confusión.
//         - Guarda el modelo entrenado (model.csv) para empaquetarlo en el móvil.
//
// Uso:
//     ./flavia_eval <ruta_dataset_flavia> [ruta_salida_model.csv]
//   donde <ruta_dataset_flavia> contiene all.csv y las imágenes (id.jpg).
// ============================================================================
#include "shape_descriptor.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs_compat {
// Pequeño helper: comprueba si un archivo existe.
inline bool exists(const std::string& p) { std::ifstream f(p); return f.good(); }
}

// Clases asignadas al Grupo G3 (Tabla 2 de la guía).
static const std::vector<int> kAssigned = {14, 15, 16, 17, 18, 19, 20};

struct Row { std::string id; int y; };

// Lee all.csv -> filas (id, y) cuyas clases estén asignadas.
std::vector<Row> readCsv(const std::string& csvPath) {
    std::vector<Row> rows;
    std::ifstream f(csvPath);
    if (!f.is_open()) { std::cerr << "No se pudo abrir " << csvPath << "\n"; return rows; }
    std::string line;
    std::getline(f, line);   // cabecera: count,id,y
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string count, id, y;
        std::getline(ss, count, ',');
        std::getline(ss, id, ',');
        std::getline(ss, y, ',');
        if (id.empty() || y.empty()) continue;
        int yi = std::stoi(y);
        if (std::find(kAssigned.begin(), kAssigned.end(), yi) != kAssigned.end())
            rows.push_back({id, yi});
    }
    return rows;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0]
                  << " <ruta_dataset_flavia> [salida_model.csv]\n";
        return 1;
    }
    std::string root = argv[1];
    std::string modelOut = (argc >= 3) ? argv[2] : "";
    std::string csvPath = root + "/all.csv";

    std::vector<Row> rows = readCsv(csvPath);
    if (rows.empty()) {
        std::cerr << "[ERROR] No hay filas para las clases asignadas o no se "
                     "encontró all.csv.\n"
                     "        Coloca el Flavia Dataset (all.csv + imágenes) en: "
                  << root << "\n";
        return 2;
    }
    std::cout << "[INFO] " << rows.size()
              << " imagenes de las clases 14..20.\n";

    // Descriptor de cada hoja, agrupado por clase.
    std::map<int, std::vector<vc::Sample>> byClass;
    int ok = 0, fail = 0;
    for (const auto& r : rows) {
        std::string imgPath = root + "/" + r.id;             // p.ej. 1300.jpg
        if (!fs_compat::exists(imgPath)) imgPath = root + "/images/" + r.id;
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
        if (img.empty()) { fail++; continue; }
        std::vector<float> d = vc::describeImage(img, /*adaptive=*/false);
        if (d.empty()) { fail++; continue; }
        byClass[r.y].push_back({d, r.y});
        ok++;
    }
    std::cout << "[INFO] descriptores OK=" << ok << " fallidos=" << fail << "\n";

    // Split 80/20 por clase.
    std::vector<vc::Sample> train, test;
    for (auto& kv : byClass) {
        auto& v = kv.second;
        size_t nTrain = static_cast<size_t>(v.size() * vc::TRAIN_RATIO);  // 80% por clase
        for (size_t i = 0; i < v.size(); ++i)
            (i < nTrain ? train : test).push_back(v[i]);
    }
    std::cout << "[INFO] train=" << train.size() << " test=" << test.size() << "\n";

    if (!modelOut.empty() && vc::saveModel(modelOut, train))
        std::cout << "[INFO] modelo guardado en " << modelOut << "\n";

    // Evaluación + matriz de confusión.
    const int C = static_cast<int>(kAssigned.size());
    std::vector<std::vector<int>> cm(C, std::vector<int>(C, 0));
    std::map<int, int> idx;
    for (int i = 0; i < C; ++i) idx[kAssigned[i]] = i;

    int correct = 0;
    for (const auto& s : test) {
        int pred = vc::classifyKNN(s.descriptor, train, vc::kKnnK);
        if (idx.count(pred)) cm[idx[s.label]][idx[pred]]++;
        if (pred == s.label) correct++;
    }
    double acc = test.empty() ? 0.0 : static_cast<double>(correct) / test.size();

    std::cout << "\n=== RESULTADOS PARTE 2 (Flavia, clases 14..20) ===\n";
    std::cout << "Accuracy = " << std::fixed << std::setprecision(4) << acc << "\n\n";
    std::cout << "Matriz de confusion (filas=real, columnas=pred):\n     ";
    for (int c : kAssigned) std::cout << std::setw(5) << c;
    std::cout << "\n";
    for (int i = 0; i < C; ++i) {
        std::cout << std::setw(4) << kAssigned[i] << " ";
        for (int j = 0; j < C; ++j) std::cout << std::setw(5) << cm[i][j];
        std::cout << "\n";
    }
    return 0;
}
