/*
 * Ruta: mobile/app/src/main/java/com/ups/vc/flavia/NativeBridge.java
 * Responsabilidad:
 *   Declarar los métodos nativos (JNI) implementados en native_lib.cpp y cargar
 *   la librería C++ 'flavia_native'. Es el único punto de contacto Java <-> C++.
 */
package com.ups.vc.flavia;

public final class NativeBridge {

    static {
        // Carga la librería nativa compilada por CMake.
        System.loadLibrary("flavia_native");
    }

    private NativeBridge() {}

    /** Carga el modelo de entrenamiento (contenido de assets/model.csv). */
    public static native int nativeInit(String modelCsv);

    /**
     * Clasifica la imagen dibujada.
     * @param matAddr dirección nativa de un cv::Mat (Utils.bitmapToMat).
     * @return texto con la especie predicha (14..20) y el descriptor.
     */
    public static native String nativeClassify(long matAddr);
}
