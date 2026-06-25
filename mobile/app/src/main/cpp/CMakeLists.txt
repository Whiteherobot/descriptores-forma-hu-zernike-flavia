# ============================================================================
# Ruta: mobile/app/src/main/cpp/CMakeLists.txt
# Responsabilidad:
#   Compilar la librería nativa de la app reutilizando EXACTAMENTE el mismo
#   núcleo C++ que la CLI de escritorio (cpp/src/shape_descriptor.cpp), más el
#   puente JNI (native_lib.cpp). Así, "los cálculos se realizan en una librería
#   nativa de C++" tal como pide la guía, y el algoritmo del móvil es idéntico
#   al validado en el PC.
# ============================================================================
cmake_minimum_required(VERSION 3.22.1)
project(flavia_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Núcleo C++ compartido (una sola fuente de verdad).
set(SHARED_CORE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../cpp)

# OpenCV vía PREFAB (lo provee el AAR org.opencv:opencv). El paquete prefab se
# llama "OpenCV" y expone el módulo/target OpenCV::opencv_java4 (headers + .so).
find_package(OpenCV REQUIRED CONFIG)

add_library(flavia_native SHARED
    native_lib.cpp
    ${SHARED_CORE}/src/shape_descriptor.cpp)

target_include_directories(flavia_native PRIVATE
    ${SHARED_CORE}/include)

find_library(log-lib log)

target_link_libraries(flavia_native
    OpenCV::opencv_java4
    ${log-lib})
