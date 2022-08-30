package com.lembergsolutions.objdetectiondemo.detection

import android.graphics.Bitmap
import java.io.Closeable

enum class HwType {
    CPU,
    GPU,
    NNAPI
}

enum class ModelType {
    Float32,
    Float16,
    Quantized
}

interface WheelDetector: Closeable {
    fun detect(bitmap: Bitmap): List<DetectionResult>
}
