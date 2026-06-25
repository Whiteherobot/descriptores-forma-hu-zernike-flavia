/*
 * Ruta: mobile/app/src/main/java/com/ups/vc/flavia/DrawView.java
 * Responsabilidad:
 *   Área de dibujo donde el usuario traza la forma de la hoja con el dedo.
 *   Mantiene un Bitmap interno (trazo NEGRO sobre fondo BLANCO) que luego se
 *   entrega al núcleo C++ para clasificar.
 */
package com.ups.vc.flavia;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class DrawView extends View {

    private Bitmap bitmap;
    private Canvas bitmapCanvas;
    private final Paint paint = new Paint();
    private final Path path = new Path();
    private float lastX, lastY;

    public DrawView(Context c, AttributeSet a) {
        super(c, a);
        paint.setColor(Color.BLACK);
        paint.setAntiAlias(true);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(12f);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
    }

    @Override
    protected void onSizeChanged(int w, int h, int ow, int oh) {
        super.onSizeChanged(w, h, ow, oh);
        bitmap = Bitmap.createBitmap(Math.max(w, 1), Math.max(h, 1), Bitmap.Config.ARGB_8888);
        bitmapCanvas = new Canvas(bitmap);
        clear();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        canvas.drawBitmap(bitmap, 0, 0, null);
        canvas.drawPath(path, paint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        float x = e.getX(), y = e.getY();
        switch (e.getAction()) {
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y);
                lastX = x; lastY = y;
                break;
            case MotionEvent.ACTION_MOVE:
                path.quadTo(lastX, lastY, (x + lastX) / 2, (y + lastY) / 2);
                lastX = x; lastY = y;
                break;
            case MotionEvent.ACTION_UP:
                bitmapCanvas.drawPath(path, paint);   // fijar el trazo en el bitmap
                path.reset();
                break;
            default:
                return false;
        }
        invalidate();
        return true;
    }

    /** Limpia el lienzo (fondo blanco). */
    public void clear() {
        if (bitmapCanvas != null) {
            bitmapCanvas.drawColor(Color.WHITE);
            path.reset();
            invalidate();
        }
    }

    /** Devuelve una copia del bitmap con el dibujo (negro sobre blanco). */
    public Bitmap getBitmap() {
        Bitmap out = bitmap.copy(Bitmap.Config.ARGB_8888, true);  // mutable: Canvas lo exige
        new Canvas(out).drawPath(path, paint);
        return out;
    }
}
