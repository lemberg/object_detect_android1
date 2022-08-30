package com.lembergsolutions.objdetectiondemo.detection

import android.content.Context

enum class ImplType {
    ML,
    ObjectDetector
}

object WheelDetectionFactory {
    fun createWheelDetector(
        context: Context,
        minimumConfidence: Float,
        implType: ImplType,
        hwType: HwType,
        modelType: ModelType,
        threads: Int
    ): WheelDetector {
        return when (implType) {
            ImplType.ML -> WheelDetectorML(context, minimumConfidence, hwType, modelType, threads)
            ImplType.ObjectDetector -> WheelDetectorObjDetector(context, minimumConfidence, hwType, modelType, threads)
        }
    }
}