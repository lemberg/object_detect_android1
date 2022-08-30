package com.lembergsolutions.objdetectiondemo.activity

import android.Manifest
import android.os.Bundle
import android.view.View
import android.widget.RadioButton
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.children
import com.google.android.material.snackbar.Snackbar
import com.lembergsolutions.objdetectiondemo.R
import com.lembergsolutions.objdetectiondemo.databinding.ActivityObjectDetectionBinding
import com.lembergsolutions.objdetectiondemo.detection.HwType
import com.lembergsolutions.objdetectiondemo.detection.ImplType
import com.lembergsolutions.objdetectiondemo.detection.ModelType
import com.lembergsolutions.objdetectiondemo.detection.WheelDetectorAnalyzer
import java.util.concurrent.Executors


class WheelsDetectionActivity : AppCompatActivity() {

    private val executor = Executors.newSingleThreadExecutor()

    private val objectDetectorConfig = WheelDetectorAnalyzer.Config(
        minimumConfidence = 0.5f,
        inputSize = 320
    )

    private val permissionsLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
        val isAllGranted = permissions.all { it.value == true }
        if (!isAllGranted) {
            showMessage(getString(R.string.permissions_missing))
        } else {
            getProcessCameraProvider(::bindCamera)
        }
    }

    private lateinit var binding: ActivityObjectDetectionBinding
    private lateinit var analyser: WheelDetectorAnalyzer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityObjectDetectionBinding.inflate(layoutInflater)
        analyser = WheelDetectorAnalyzer(applicationContext, objectDetectorConfig, ::onDetectionResult)
        setContentView(binding.root)
        initViews()
        checkPermissions()
    }

    override fun onStart() {
        super.onStart()
        analyser.start()
    }

    override fun onStop() {
        analyser.close()
        super.onStop()
    }

    override fun onDestroy() {
        executor.shutdown()
        super.onDestroy()
    }

    private fun initViews() {
        binding.includeButtons.radioGroupImplType.setOnCheckedChangeListener { _, checkedId ->
            val newImplType = when (checkedId) {
                R.id.radio_ml -> ImplType.ML
                R.id.radio_obj_detect -> ImplType.ObjectDetector
                else -> error("Unexpected")
            }
            analyser.implType = newImplType
        }

        binding.includeButtons.radioGroupHwType.setOnCheckedChangeListener { _, checkedId ->
            val newHwType = when (checkedId) {
                R.id.radio_cpu -> HwType.CPU
                R.id.radio_gpu -> HwType.GPU
                R.id.radio_nnapi -> HwType.NNAPI
                else -> error("Unexpected")
            }
            analyser.hwType = newHwType
        }

        binding.includeButtons.radioGroupModelType.setOnCheckedChangeListener { _, checkedId ->
            val newModelType = when (checkedId) {
                R.id.radio_float32_model -> ModelType.Float32
                R.id.radio_float16_model -> ModelType.Float16
                R.id.radio_quant_model -> ModelType.Quantized
                else -> error("Unexpected")
            }
            analyser.modelType = newModelType
        }

        binding.includeButtons.radioThread1.setOnClickListener(::onThreadsClicked)
        binding.includeButtons.radioThread2.setOnClickListener(::onThreadsClicked)
        binding.includeButtons.radioThread3.setOnClickListener(::onThreadsClicked)
        binding.includeButtons.radioThread4.setOnClickListener(::onThreadsClicked)
    }

    private fun onThreadsClicked(view: View) {
        // clear previous check
        binding.includeButtons.gridThreads.children.forEach {
            (it as? RadioButton)?.isChecked = false
        }
        // set new check
        (view as? RadioButton)?.isChecked = true

        val threads = when (view.id) {
            R.id.radio_thread_1 -> 1
            R.id.radio_thread_2 -> 2
            R.id.radio_thread_3 -> 3
            R.id.radio_thread_4 -> 4
            else -> error("Unexpected")
        }

        analyser.threads = threads
    }

    private fun checkPermissions() {
        permissionsLauncher.launch(arrayOf(
            Manifest.permission.CAMERA
        ))
    }

    private fun bindCamera(cameraProvider: ProcessCameraProvider) {
        val preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .build()

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//            .setOutputImageRotationEnabled(true)
//            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalysis.setAnalyzer(
            executor,
            analyser
        )

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        cameraProvider.unbindAll()

        cameraProvider.bindToLifecycle(
            this,
            cameraSelector,
            imageAnalysis,
            preview
        )

        preview.setSurfaceProvider(binding.previewView.surfaceProvider)
    }

    private fun getProcessCameraProvider(onDone: (ProcessCameraProvider) -> Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(
            { onDone.invoke(cameraProviderFuture.get()) },
            ContextCompat.getMainExecutor(this)
        )
    }

    private fun onDetectionResult(result: WheelDetectorAnalyzer.Result) {
        runOnUiThread {
            binding.resultOverlay.updateResults(result)
        }
    }

    private fun showMessage(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
    }
}