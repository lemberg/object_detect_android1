package com.lembergsolutions.objdetectiondemo.detection

import android.graphics.RectF

data class DetectionResult(
        val title: String,
        val confidence: Float,
        val location: RectF
) {
    val text: String by lazy {
        "$title[${"%.2f".format(confidence)}]"
    }
}