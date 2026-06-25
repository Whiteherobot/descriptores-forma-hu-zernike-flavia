if(NOT TARGET OpenCV::opencv_java4)
add_library(OpenCV::opencv_java4 SHARED IMPORTED)
set_target_properties(OpenCV::opencv_java4 PROPERTIES
    IMPORTED_LOCATION "/home/ing_michi/.gradle/caches/transforms-4/cb12912c993d18f5db48d2f81d4ee35b/transformed/opencv-4.11.0/prefab/modules/opencv_java4/libs/android.x86_64/libopencv_java4.so"
    INTERFACE_INCLUDE_DIRECTORIES "/home/ing_michi/.gradle/caches/transforms-4/cb12912c993d18f5db48d2f81d4ee35b/transformed/opencv-4.11.0/prefab/modules/opencv_java4/include"
    INTERFACE_LINK_LIBRARIES ""
)
endif()

