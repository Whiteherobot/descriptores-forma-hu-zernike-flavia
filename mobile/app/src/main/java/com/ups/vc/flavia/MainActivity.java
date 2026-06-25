/*
 * Ruta: mobile/app/src/main/java/com/ups/vc/flavia/MainActivity.java
 * Responsabilidad:
 *   Pantalla principal de la app (Parte 2). Presenta el área de dibujo y el
 *   botón "Clasificar forma". Al pulsarlo:
 *     1. Toma el bitmap del DrawView.
 *     2. Lo convierte a cv::Mat con OpenCV (Utils.bitmapToMat).
 *     3. Llama al núcleo C++ (NativeBridge.nativeClassify) que calcula el
 *        descriptor de Fourier y devuelve la especie (14..20).
 *     4. Muestra la etiqueta y el descriptor en pantalla.
 *   El modelo (descriptores de entrenamiento) se carga una vez desde
 *   assets/model.csv.
 */
package com.ups.vc.flavia;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public class MainActivity extends AppCompatActivity {

    private DrawView drawView;
    private TextView resultText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Inicializar OpenCV (capa Java). Necesario antes de usar Mat/Utils.
        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "No se pudo inicializar OpenCV", Toast.LENGTH_LONG).show();
        }

        drawView = findViewById(R.id.draw_view);
        resultText = findViewById(R.id.result_text);
        Button classifyBtn = findViewById(R.id.btn_classify);
        Button clearBtn = findViewById(R.id.btn_clear);

        // Cargar el modelo entrenado desde los assets hacia el núcleo C++.
        int n = NativeBridge.nativeInit(loadAsset("model.csv"));
        resultText.setText(n > 0
                ? "Modelo cargado (" + n + " muestras). Dibuja una hoja."
                : "Falta assets/model.csv. Genera el modelo con run_part2.py.");

        classifyBtn.setOnClickListener(v -> classify());
        clearBtn.setOnClickListener(v -> {
            drawView.clear();
            resultText.setText("Lienzo limpio. Dibuja una hoja.");
        });
    }

    private void classify() {
        Bitmap bmp = drawView.getBitmap();
        Mat mat = new Mat();
        Utils.bitmapToMat(bmp, mat);            // BGRA en memoria nativa
        String result = NativeBridge.nativeClassify(mat.getNativeObjAddr());
        mat.release();
        resultText.setText(result);
    }

    /** Lee un archivo de assets/ como String. */
    private String loadAsset(String name) {
        StringBuilder sb = new StringBuilder();
        try (InputStream is = getAssets().open(name);
             BufferedReader br = new BufferedReader(
                     new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line).append('\n');
        } catch (Exception e) {
            return "";   // el modelo aún no se ha empaquetado
        }
        return sb.toString();
    }
}
