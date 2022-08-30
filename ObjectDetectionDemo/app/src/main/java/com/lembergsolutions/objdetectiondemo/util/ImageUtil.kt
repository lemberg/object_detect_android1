package com.lembergsolutions.objdetectiondemo.util

import android.graphics.Matrix
import kotlin.math.abs

object ImageUtil {

    fun getTransformMatrix(
            applyRotation: Int,
            srcWidth: Int, srcHeight: Int,
            dstWidth: Int, dstHeight: Int
    ): Matrix {
        val matrix = Matrix()

        if (applyRotation != 0) {
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f)
            matrix.postRotate(applyRotation.toFloat())
        }

        val transpose = (abs(applyRotation) + 90) % 180 == 0

        val inWidth = if (transpose) srcHeight else srcWidth
        val inHeight = if (transpose) srcWidth else srcHeight

        if (inWidth != dstWidth || inHeight != dstHeight) {
            val scaleFactorX = dstWidth / inWidth.toFloat()
            val scaleFactorY = dstHeight / inHeight.toFloat()

            matrix.postScale(scaleFactorX, scaleFactorY)
        }

        if (applyRotation != 0) {
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f)
        }

        return matrix
    }
}