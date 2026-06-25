// ============================================================================
// Ruta: mobile/app/src/main/cpp/native_lib.cpp
// Responsabilidad:
//   Puente JNI entre la capa Java (UI) y el núcleo C++ de visión. Expone:
//     - nativeInit(modelCsv)  : carga el modelo (descriptores de entrenamiento).
//     - nativeClassify(matAddr): recibe la imagen dibujada (cv::Mat), calcula el
//       descriptor de Fourier de coordenadas complejas y devuelve la especie
//       predicha (14..20) junto con los primeros valores del descriptor.
//
//   Toda la matemática vive en shape_descriptor.cpp (compartido con la CLI).
// ============================================================================
#include <jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "shape_descriptor.h"

#define LOG_TAG "FlaviaNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// Modelo de entrenamiento (se carga una vez desde los assets).
static std::vector<vc::Sample> g_model;

extern "C" {

// Carga el modelo desde el contenido de model.csv (label,d0,d1,...).
JNIEXPORT jint JNICALL
Java_com_ups_vc_flavia_NativeBridge_nativeInit(JNIEnv* env, jclass, jstring modelCsv) {
    const char* csv = env->GetStringUTFChars(modelCsv, nullptr);
    g_model.clear();
    std::stringstream all(csv);
    std::string line;
    while (std::getline(all, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ',');
        vc::Sample s;
        s.label = std::stoi(tok);
        while (std::getline(ss, tok, ',')) s.descriptor.push_back(std::stof(tok));
        g_model.push_back(std::move(s));
    }
    env->ReleaseStringUTFChars(modelCsv, csv);
    LOGI("Modelo cargado: %zu muestras", g_model.size());
    return static_cast<jint>(g_model.size());
}

// Clasifica el dibujo. matAddr es la dirección de un cv::Mat (BGRA/Gray) creado
// en Java con OpenCV (Utils.bitmapToMat). Devuelve un texto con la etiqueta y el
// descriptor para mostrarlo en pantalla.
JNIEXPORT jstring JNICALL
Java_com_ups_vc_flavia_NativeBridge_nativeClassify(JNIEnv* env, jclass, jlong matAddr) {
    cv::Mat& img = *reinterpret_cast<cv::Mat*>(matAddr);

    std::ostringstream out;
    try {
        // El dibujo viene como trazo oscuro sobre fondo claro -> umbral adaptativo.
        std::vector<float> desc = vc::describeImage(img, /*useAdaptiveThreshold=*/true);

        if (desc.empty()) {
            out << "No se detecto forma. Dibuja una hoja y reintenta.";
        } else if (g_model.empty()) {
            out << "Modelo no cargado (assets/model.csv).";
        } else {
            int label = vc::classifyKNN(desc, g_model, vc::kKnnK);
            out << "ESPECIE CLASIFICADA: " << label << "\n";
            out << "Descriptor de Fourier:\n";
            for (size_t i = 0; i < desc.size() && i < 6; ++i) {
                out << (i ? ", " : "") << desc[i];
            }
            out << ", ...";
        }
    } catch (const std::exception& e) {
        LOGI("Error al clasificar: %s", e.what());
        out.str("");
        out << "Error al procesar la imagen: " << e.what();
    }
    return env->NewStringUTF(out.str().c_str());
}

}  // extern "C"
