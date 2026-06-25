# ============================================================================
# Ruta:           cpp/CMakeLists.txt
# Responsabilidad:
#     Compilar el núcleo C++ de la Parte 2 para ESCRITORIO (validación):
#         - libshape_core   : el algoritmo reutilizable (mismo .cpp que Android).
#         - flavia_eval      : CLI de entrenamiento/evaluación sobre Flavia.
#         - selftest         : prueba sin dataset (formas sintéticas).
#
# Uso:
#     cd cpp && cmake -B build && cmake --build build -j
#     ./build/selftest
#     ./build/flavia_eval <ruta_flavia> model.csv
# ============================================================================
cmake_minimum_required(VERSION 3.16)
project(vc_shape_part2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sólo los módulos necesarios (evita arrastrar viz/hdf con dependencias rotas).
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Biblioteca reutilizable con el algoritmo (la misma que usa la app Android).
add_library(shape_core STATIC src/shape_descriptor.cpp)
target_include_directories(shape_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                             ${OpenCV_INCLUDE_DIRS})
target_link_libraries(shape_core PUBLIC opencv_core opencv_imgproc opencv_imgcodecs)

# CLI de evaluación sobre el Flavia Dataset.
add_executable(flavia_eval apps/main.cpp)
target_link_libraries(flavia_eval PRIVATE shape_core)

# Prueba autónoma (sin dataset).
add_executable(selftest apps/selftest.cpp)
target_link_libraries(selftest PRIVATE shape_core)
