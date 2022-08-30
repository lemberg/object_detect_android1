package com.lembergsolutions.objdetectiondemo.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector


class WheelDetectorObjDetector constructor(
    context: Context,
    minimumConfidence: Float,
    hwType: HwType,
    modelType: ModelType,
    threads: Int
): WheelDetector {
    private val objectDetector = ObjectDetector.createFromFileAndOptions(
        context,
        getModelFileName(modelType),
        ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(1)
            .setScoreThreshold(minimumConfidence)
            .setBaseOptions(buildBaseOptions(hwType, threads))
            .build()
    )

    override fun detect(bitmap: Bitmap): List<DetectionResult> {
        val image = TensorImage.fromBitmap(bitmap)
        val results = objectDetector.detect(image)
        return convertDetectionResult(results, image.width, image.height)
    }

    override fun close() {
        objectDetector.close()
    }

    private fun convertDetectionResult(list: List<Detection>, width: Int, height: Int): List<DetectionResult> {
        return list.map {
            val category = it.categories.maxByOrNull { category -> category.score } ?: return@map null
            val title = category.displayName
            val location = RectF(
                it.boundingBox.left / width,
                it.boundingBox.top / height,
                it.boundingBox.right / width,
                it.boundingBox.bottom / height
            )
            DetectionResult(title, category.score, location)
        }.filterNotNull()
    }

    companion object {
        private fun getModelFileName(modelType: ModelType): String {
            return when (modelType) {
                ModelType.Float32 -> "wheels_detection.tflite"
                ModelType.Float16 -> "wheels_detection_fp16.tflite"
                ModelType.Quantized -> "wheels_detection_quant.tflite"
            }
        }

        private fun buildBaseOptions(hwType: HwType, threads: Int): BaseOptions {
            return when (hwType) {
                HwType.CPU -> BaseOptions.builder().setNumThreads(threads).build()
                HwType.GPU -> BaseOptions.builder().useGpu().setNumThreads(threads).build()
                HwType.NNAPI -> BaseOptions.builder().useNnapi().setNumThreads(threads).build()
            }
        }
    }
}