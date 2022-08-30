package com.lembergsolutions.objdetectiondemo.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.media.Image
import android.os.SystemClock
import androidx.annotation.GuardedBy
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.lembergsolutions.objdetectiondemo.util.ImageUtil
import com.lembergsolutions.objdetectiondemo.util.YuvToRgbConverter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder


class WheelDetectorAnalyzer(
    private val context: Context,
    private val config: Config,
    private val onDetectionResult: (Result) -> Unit
) : ImageAnalysis.Analyzer, Closeable {

    data class Config(
        val minimumConfidence: Float,
        val inputSize: Int
    )

    data class Result(
        val objects: List<DetectionResult>,
        val imageWidth: Int,
        val imageHeight: Int,
        val prepareTime: Long,
        val tensorflowTime: Long
    )

    private val yuvToRgbConverter = YuvToRgbConverter(context)

    private val wheelDetectorLock = Any()
    @GuardedBy("wheelDetectorLock")
    private var wheelDetector: WheelDetector? = null
    private var isStopped = true
    private var rgbBitmap: Bitmap? = null
    private val resizedBitmap = Bitmap.createBitmap(config.inputSize, config.inputSize, Bitmap.Config.ARGB_8888)
    private var matrixToInput: Matrix? = null

    var implType: ImplType = ImplType.ML
        set(value) {
            field = value
            closeWheelDetector()
        }

    var hwType: HwType = HwType.CPU
        set(value) {
            field = value
            closeWheelDetector()
        }

    var modelType: ModelType = ModelType.Float32
        set(value) {
            field = value
            closeWheelDetector()
        }

    var threads: Int = 1
        set(value) {
            field = value
            closeWheelDetector()
        }

    fun start() {
        isStopped = false
    }

    override fun close() {
        isStopped = true
        closeWheelDetector()
    }

    private fun closeWheelDetector() {
        synchronized(wheelDetectorLock) {
            wheelDetector?.close()
            wheelDetector = null
        }
    }

    private fun getWheelDetector(): WheelDetector? {
        return wheelDetector ?: createWheelDetector()?.apply {
            wheelDetector = this
        }
    }

    private fun createWheelDetector(): WheelDetector? {
        if (isStopped) return null
        return WheelDetectionFactory.createWheelDetector(context, config.minimumConfidence, implType, hwType, modelType, threads)
    }

    override fun analyze(image: ImageProxy) {
        synchronized(wheelDetectorLock) {
            image.use {
                val wheelDetector = getWheelDetector() ?: return

                val startTime = SystemClock.uptimeMillis()
                val rotationDegrees = image.imageInfo.rotationDegrees
                val transformation = getTransformation(rotationDegrees, image.width, image.height)

                val rgbBitmap = getArgbBitmap(image.width, image.height)
                yuvToRgbConverter.yuvToRgb(image, rgbBitmap)

                Canvas(resizedBitmap).drawBitmap(rgbBitmap, transformation, null)

                val prepareEndTime = SystemClock.uptimeMillis()

                val detectedObjs: List<DetectionResult> = wheelDetector.detect(resizedBitmap)
                val detectEndTime = SystemClock.uptimeMillis()

                val result = Result(
                    detectedObjs,
                    config.inputSize,
                    config.inputSize,
                    prepareEndTime - startTime,
                    detectEndTime - prepareEndTime
                )

                onDetectionResult(result)
            }
        }
    }

    private fun getTransformation(rotationDegrees: Int, srcWidth: Int, srcHeight: Int): Matrix {
        return matrixToInput ?: ImageUtil.getTransformMatrix(rotationDegrees, srcWidth, srcHeight, config.inputSize, config.inputSize).apply {
            matrixToInput = this
        }
    }

    private fun getArgbBitmap(width: Int, height: Int): Bitmap {
        return rgbBitmap ?: Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            rgbBitmap = this
        }
    }
}