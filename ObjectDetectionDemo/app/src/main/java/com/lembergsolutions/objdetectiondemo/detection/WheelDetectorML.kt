package com.lembergsolutions.objdetectiondemo.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import com.lembergsolutions.objdetectiondemo.ml.WheelsDetection
import com.lembergsolutions.objdetectiondemo.ml.WheelsDetectionFp16
import com.lembergsolutions.objdetectiondemo.ml.WheelsDetectionQuant
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.model.Model


class WheelDetectorML constructor(
    context: Context,
    private val minimumConfidence: Float,
    hwType: HwType,
    modelType: ModelType,
    threads: Int
): WheelDetector {

    private val model: Any = createModel(context, hwType, modelType, threads)

    override fun detect(bitmap: Bitmap): List<DetectionResult> {
        val image = TensorImage.fromBitmap(bitmap)
        return processModel(image)
    }

    override fun close() {
        when (model) {
            is WheelsDetection -> model.close()
            is WheelsDetectionFp16 -> model.close()
            is WheelsDetectionQuant -> model.close()
        }
    }

    private fun processModel(image: TensorImage): List<DetectionResult> {
        return when (model) {
            is WheelsDetection -> convertDetectionResult(model.process(image).detectionResultList, image.width, image.height)
            is WheelsDetectionFp16 -> convertDetectionResultFp16(model.process(image).detectionResultList, image.width, image.height)
            is WheelsDetectionQuant -> convertDetectionResultQuant(model.process(image).detectionResultList, image.width, image.height)
            else -> error("Unexpected")
        }
    }

    private fun convertDetectionResultQuant(list: List<WheelsDetectionQuant.DetectionResult>, width: Int, height: Int): List<DetectionResult> {
        return list.filter { it.scoreAsFloat >= minimumConfidence }.map {
            val title = it.categoryAsString
            val rect = it.locationAsRectF
            val location = RectF(
                rect.left / width,
                rect.top / height,
                rect.right / width,
                rect.bottom / height
            )
            DetectionResult(title, it.scoreAsFloat, location)
        }
    }

    private fun convertDetectionResultFp16(list: List<WheelsDetectionFp16.DetectionResult>, width: Int, height: Int): List<DetectionResult> {
        return list.filter { it.scoreAsFloat >= minimumConfidence }.map {
            val title = it.categoryAsString
            val rect = it.locationAsRectF
            val location = RectF(
                rect.left / width,
                rect.top / height,
                rect.right / width,
                rect.bottom / height
            )
            DetectionResult(title, it.scoreAsFloat, location)
        }
    }

    private fun convertDetectionResult(list: List<WheelsDetection.DetectionResult>, width: Int, height: Int): List<DetectionResult> {
        return list.filter { it.scoreAsFloat >= minimumConfidence }.map {
            val title = it.categoryAsString
            val rect = it.locationAsRectF
            val location = RectF(
                rect.left / width,
                rect.top / height,
                rect.right / width,
                rect.bottom / height
            )
            DetectionResult(title, it.scoreAsFloat, location)
        }
    }

    companion object {
        private fun convertHwType(hwType: HwType): Model.Device {
            return when (hwType) {
                HwType.CPU -> Model.Device.CPU
                HwType.GPU -> Model.Device.GPU
                HwType.NNAPI -> Model.Device.NNAPI
            }
        }

        private fun createModel(context: Context, hwType: HwType, modelType: ModelType, threads: Int): Any {
            return when (modelType) {
                ModelType.Float32 -> WheelsDetection.newInstance(context,
                    Model.Options.Builder()
                        .setNumThreads(threads)
                        .setDevice(convertHwType(hwType))
                        .build())
                ModelType.Float16 -> WheelsDetectionFp16.newInstance(context,
                    Model.Options.Builder()
                        .setNumThreads(threads)
                        .setDevice(convertHwType(hwType))
                        .build())
                ModelType.Quantized -> WheelsDetectionQuant.newInstance(context,
                    Model.Options.Builder()
                        .setNumThreads(threads)
                        .setDevice(convertHwType(hwType))
                        .build())
            }
        }
    }
}