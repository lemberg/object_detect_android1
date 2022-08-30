package com.lembergsolutions.objdetectiondemo.view

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.lembergsolutions.objdetectiondemo.R
import com.lembergsolutions.objdetectiondemo.detection.WheelDetectorAnalyzer
import java.text.DecimalFormat


class DetectionResultOverlay @JvmOverloads constructor(
        context: Context,
        attrs: AttributeSet? = null,
        defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
    private val FILTER_ALPHA = 0.5f
    private val sb = StringBuilder(100)
    private val numberFormatter = DecimalFormat("#.00")

    private var prepareTime = 0L
    private var tensorflowTime = 0L

    private val boxPaint = Paint().apply {
        color = resources.getColor(R.color.result_overlay_color)
        style = Paint.Style.STROKE
        strokeWidth = resources.getDimension(R.dimen.result_overlay_stroke_width)
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = resources.getColor(R.color.result_overlay_color)
        textSize = resources.getDimension(R.dimen.result_overlay_text_size)
    }

    private var result: WheelDetectorAnalyzer.Result? = null

    fun updateResults(result: WheelDetectorAnalyzer.Result) {
        this.result = result
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val result = result ?: return

        val scaleFactorX = measuredWidth / result.imageWidth.toFloat()
        val scaleFactorY = measuredHeight / result.imageHeight.toFloat()

        result.objects.forEach { obj ->
            val left = obj.location.left * result.imageWidth * scaleFactorX
            val top = obj.location.top * result.imageHeight * scaleFactorY
            val right = obj.location.right * result.imageWidth * scaleFactorX
            val bottom = obj.location.bottom * result.imageHeight * scaleFactorY

            canvas.drawRect(left, top, right, bottom, boxPaint)
            canvas.drawText(obj.text, left, top - textPaint.textSize, textPaint)
        }

        prepareTime += ((result.prepareTime - prepareTime) * FILTER_ALPHA).toLong()
        tensorflowTime += ((result.tensorflowTime - tensorflowTime) * FILTER_ALPHA).toLong()
        drawFps(canvas)
    }

    private fun drawFps(canvas: Canvas) {
        val str = getFpsText()
        canvas.drawText(str, 0f, canvas.height - textPaint.textSize, textPaint)
    }

    private fun getFpsText(): String {
        val fps = if (tensorflowTime != 0L) 1000f / tensorflowTime else 0f

        sb.setLength(0)
        sb.append("FPS: ")
        if (fps < 1) sb.append(numberFormatter.format(fps))
        else sb.append(fps.toInt())

        return sb.toString()
    }
}