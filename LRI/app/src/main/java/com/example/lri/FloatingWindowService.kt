package com.example.lri

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import android.graphics.*
import android.media.MediaMetadataRetriever
import android.os.*
import android.util.Log
import android.widget.Toast
import androidx.camera.core.CameraSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.core.util.Consumer
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.lifecycle.LifecycleService
import com.github.only52607.compose.window.ComposeFloatingWindow

// 加入 MediaPipe 相關依賴
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker

import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream

class FloatingWindowService : LifecycleService() {

    private lateinit var floatingWindow: ComposeFloatingWindow
    private lateinit var customLifecycleOwner: CustomLifecycleOwner
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    private var cameraProvider: ProcessCameraProvider? = null

    // 替換為 MediaPipe 的 FaceLandmarker
    private lateinit var faceLandmarker: FaceLandmarker

    private val serviceScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var recordingStartTime: Long = 0L
    private val splitTimestamps = mutableListOf<Long>()

    private val TARGET_FRAMES = 40
    private val TARGET_FPS = 25

    // ---- 端到端延遲量測 ----
    // 量測時務必設為 false：把 40 張 PNG 寫入外部儲存的時間並非管線的一部分
    private val SAVE_DEBUG_FRAMES = false
    private var tExtractNs = 0L      // 影格抽取
    private var tDetectNs = 0L       // MediaPipe 唇部偵測
    private var tCropNs = 0L         // 裁切 + 灰階 + 縮放
    private var tRecognizeNs = 0L    // 模型推論 + 字典解碼
    private var nFrames = 0

    private lateinit var videoOutputFile: File

    // ✅ 恢復真實模型宣告
    private lateinit var lipManager: LipReadingManager
    private var updateUICallback: ((List<String>) -> Unit)? = null

    companion object {
        private const val NOTIFICATION_ID = 101
        private const val NOTIFICATION_CHANNEL_ID = "LipReadingInputChannel"
    }

    private class CustomLifecycleOwner : LifecycleOwner {
        private val lifecycleRegistry = LifecycleRegistry(this)
        init { lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_CREATE) }
        override val lifecycle: Lifecycle get() = lifecycleRegistry
        fun start() { lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_START) }
        fun destroy() { lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_DESTROY) }
    }

    override fun onBind(intent: Intent): IBinder? {
        super.onBind(intent)
        return null
    }

    override fun onCreate() {
        super.onCreate()
        customLifecycleOwner = CustomLifecycleOwner()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification())
        customLifecycleOwner.start()

        // 解決 Input Latency：把 MediaPipe 也丟到背景執行
        serviceScope.launch(Dispatchers.IO) {
            initializeFaceLandmarker()
        }

        initCameraProvider()

        // ✅ 解除封印：載入真實的 LipReadingManager 與模型字典
        lipManager = LipReadingManager(applicationContext)

        floatingWindow = ComposeFloatingWindow(applicationContext)
        floatingWindow.setContent {
            var isExpanded by remember { mutableStateOf(false) }
            var showCandidates by remember { mutableStateOf(false) }
            var statusText by remember { mutableStateOf<String>("") }
            var candidateList by remember { mutableStateOf<List<String>>(emptyList()) }

            DisposableEffect(Unit) {
                updateUICallback = { results ->
                    if (results.isNotEmpty()) {
                        statusText = results[0]
                        candidateList = results
                        showCandidates = true
                    } else {
                        statusText = "無法辨識"
                        showCandidates = false
                    }
                }
                onDispose { updateUICallback = null }
            }

            LaunchedEffect(isExpanded) {
                if (isExpanded) bindCamera() else unbindCamera()
            }

            Row(verticalAlignment = Alignment.Top) {
                Box {
                    if (!isExpanded) {
                        CollapsedView(onExpand = { isExpanded = true })
                    } else {
                        ExpandedView(
                            onCollapse = { isExpanded = false },
                            onStartRecord = {
                                startRecording()
                                showCandidates = false
                                statusText = ""
                            },
                            onStopRecord = {
                                stopRecording {
                                    showCandidates = true
                                    statusText = "影像處理中..."
                                    candidateList = listOf("分析中...")
                                }
                            },
                            onWordSplit = { recordWordSplit() }
                        )
                    }
                }
                if (showCandidates) {
                    if (statusText.contains("處理中") || statusText.contains("分析中") || statusText.contains("無法辨識") || statusText.contains("失敗")) {
                        CandidateButtons(candidates = listOf(statusText), onCandidateSelected = {})
                    } else {
                        CandidateButtons(
                            candidates = candidateList,
                            onCandidateSelected = { text ->
                                showCandidates = false
                                MyAccessibilityService.pasteTextFlow.tryEmit(text)
                            }
                        )
                    }
                }
            }
        }
        floatingWindow.show()
    }

    override fun onDestroy() {
        super.onDestroy()
        try { activeRecording?.stop() } catch (e: Exception) {}
        unbindCamera()

        // 解決 Lifecycle Stability：所有的 lateinit 都要加上 isInitialized 保護
        if (::faceLandmarker.isInitialized) {
            try { faceLandmarker.close() } catch (e: Exception) {}
        }

        if (::customLifecycleOwner.isInitialized) {
            customLifecycleOwner.destroy()
        }

        if (::floatingWindow.isInitialized) {
            floatingWindow.hide()
        }

        if (::lipManager.isInitialized) {
            lipManager.close()
        }

        serviceScope.cancel()
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    // MediaPipe 初始化邏輯
    private fun initializeFaceLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()

        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(1)
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(applicationContext, options)
    }

    private fun initCameraProvider() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
            } catch (e: Exception) { Log.e("CameraX", "Provider Init Failed", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCamera() {
        val provider = cameraProvider ?: return
        val qualitySelector = QualitySelector.fromOrderedList(
            listOf(Quality.SD, Quality.LOWEST),
            FallbackStrategy.lowerQualityOrHigherThan(Quality.SD)
        )
        val recorder = Recorder.Builder().setQualitySelector(qualitySelector).build()
        videoCapture = VideoCapture.withOutput(recorder)

        try {
            provider.unbindAll()
            provider.bindToLifecycle(customLifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, videoCapture)
        } catch (exc: Exception) { Log.e("CameraX", "相機綁定失敗", exc) }
    }

    private fun unbindCamera() {
        try { cameraProvider?.unbindAll() } catch (e: Exception) {}
    }

    @SuppressLint("MissingPermission")
    private fun startRecording() {
        val localVideoCapture = videoCapture ?: return
        val outputDirectory = File(getExternalFilesDir(Environment.DIRECTORY_MOVIES), "LRI_Raw_Videos")
        outputDirectory.mkdirs()
        videoOutputFile = File(outputDirectory, "temp_recording.mp4")
        if (videoOutputFile.exists()) videoOutputFile.delete()

        splitTimestamps.clear()
        recordingStartTime = System.currentTimeMillis()

        activeRecording = localVideoCapture.output
            .prepareRecording(this, FileOutputOptions.Builder(videoOutputFile).build())
            .start(ContextCompat.getMainExecutor(this), videoRecordEventConsumer)

        vibrate()
        Toast.makeText(this, "開始錄影", Toast.LENGTH_SHORT).show()
    }

    private fun recordWordSplit() {
        if (recordingStartTime == 0L) return
        val relativeTimeUs = (System.currentTimeMillis() - recordingStartTime) * 1000
        splitTimestamps.add(relativeTimeUs)
        vibrate()
    }

    private fun stopRecording(onProcessingStart: () -> Unit) {
        if (activeRecording != null) {
            vibrate()
            val endTimeUs = (System.currentTimeMillis() - recordingStartTime) * 1000
            splitTimestamps.add(endTimeUs)
            activeRecording?.stop()
            activeRecording = null
            recordingStartTime = 0L
            onProcessingStart()
        }
    }

    private val videoRecordEventConsumer = Consumer<VideoRecordEvent> { event ->
        if (event is VideoRecordEvent.Finalize) {
            if (!event.hasError()) {
                processRecordedVideo(videoOutputFile)
            } else {
                serviceScope.launch(Dispatchers.Main) {
                    updateUICallback?.invoke(listOf("錄影失敗"))
                }
            }
        }
    }

    // ✅ 真實推論與處理邏輯 (已加入完整 Bitmap 回收機制)
    private fun processRecordedVideo(videoFile: File) {
        serviceScope.launch(Dispatchers.IO) {
            val rootDir = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "LRI_Processed_Frames")
            if (rootDir.exists()) { rootDir.deleteRecursively() }
            rootDir.mkdirs()
            val startTime = System.currentTimeMillis()
            tExtractNs = 0L; tDetectNs = 0L; tCropNs = 0L; tRecognizeNs = 0L; nFrames = 0
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(videoFile.absolutePath)
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { updateUICallback?.invoke(listOf("無法讀取影片檔")) }
                return@launch
            }

            val allWordsBitmaps = mutableListOf<List<Bitmap>>()
            var startUs: Long = 0L

            splitTimestamps.forEachIndexed { index, endUs ->
                val segmentDurationUs = endUs - startUs
                if (segmentDurationUs > 100000) {

                    val stepUs = 1000000L / TARGET_FPS
                    var lastKnownRect: Rect? = null
                    val currentWordFrames = mutableListOf<Bitmap>()
                    var targetTimeUs = startUs

                    while (targetTimeUs <= endUs && currentWordFrames.size < TARGET_FRAMES) {

                        val tExtract0 = System.nanoTime()
                        var rawFrame = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                            retriever.getScaledFrameAtTime(targetTimeUs, MediaMetadataRetriever.OPTION_CLOSEST, 360, 640)
                        } else {
                            retriever.getFrameAtTime(targetTimeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                        }
                        // MediaPipe 只接受 ARGB_8888。部分機型（例如 Xperia 10 III）的解碼器
                        // 會回傳 RGB_565 或 HARDWARE，直接送進 BitmapImageBuilder 會拋
                        // UnsupportedOperationException，導致每一格都退回黑畫面。
                        if (rawFrame != null && rawFrame.config != Bitmap.Config.ARGB_8888) {
                            val converted = rawFrame.copy(Bitmap.Config.ARGB_8888, false)
                            rawFrame.recycle()
                            rawFrame = converted
                        }
                        tExtractNs += System.nanoTime() - tExtract0

                        if (rawFrame != null) {
                            nFrames++
                            val tDetect0 = System.nanoTime()
                            val detectedRect = detectFaceAndGetRect(rawFrame)
                            tDetectNs += System.nanoTime() - tDetect0
                            if (detectedRect != null) {
                                lastKnownRect = detectedRect
                            }

                            val finalFrame = if (lastKnownRect != null) {
                                var left = lastKnownRect.left
                                var top = lastKnownRect.top
                                var width = lastKnownRect.width()
                                var height = lastKnownRect.height()

                                if (left < 0) left = 0
                                if (top < 0) top = 0
                                if (left + width > rawFrame.width) width = rawFrame.width - left
                                if (top + height > rawFrame.height) height = rawFrame.height - top

                                val tCrop0 = System.nanoTime()
                                val cropped = Bitmap.createBitmap(rawFrame, left, top, width, height)
                                val resized = toGrayscaleAndResize(cropped, 88, 88)

                                // ✅ 記憶體優化 1：立刻回收過渡期的臉部裁切圖
                                if (cropped != rawFrame) {
                                    cropped.recycle()
                                }
                                tCropNs += System.nanoTime() - tCrop0

                                resized
                            } else {
                                val black = Bitmap.createBitmap(88, 88, Bitmap.Config.ARGB_8888)
                                black.eraseColor(Color.BLACK)
                                black
                            }
                            currentWordFrames.add(finalFrame)

                            // ✅ 記憶體優化 2：立刻回收 MediaMetadataRetriever 吐出來的 360x640 原始大圖
                            rawFrame.recycle()
                        }
                        targetTimeUs += stepUs
                    }

                    if (currentWordFrames.isNotEmpty() && currentWordFrames.size < TARGET_FRAMES) {
                        val padCount = TARGET_FRAMES - currentWordFrames.size
                        val padFrame = currentWordFrames.last()
                        for (i in 0 until padCount) {
                            currentWordFrames.add(padFrame)
                        }
                    }

                    if (currentWordFrames.isNotEmpty()) {
                        if (SAVE_DEBUG_FRAMES) {
                            val wordDir = File(rootDir, "word_${index + 1}")
                            wordDir.mkdirs()

                            for (i in currentWordFrames.indices) {
                                val frameFile = File(wordDir, "frame_${String.format("%02d", i)}.png")
                                saveBitmap(currentWordFrames[i], frameFile)
                            }
                        }

                        allWordsBitmaps.add(currentWordFrames)
                    }
                }
                startUs = endUs
            }
            retriever.release()

            // 執行模型推論
            if (allWordsBitmaps.isNotEmpty()) {
                val tRec0 = System.nanoTime()
                val results = lipManager.recognizeSentence(allWordsBitmaps)
                tRecognizeNs = System.nanoTime() - tRec0

                logStageLatency(allWordsBitmaps.size, System.currentTimeMillis() - startTime)

                withContext(Dispatchers.Main) {
                    updateUICallback?.invoke(results)
                }

                // ✅ 記憶體優化 3：推論結束後，將存放在陣列裡的 88x88 小圖全數銷毀
                allWordsBitmaps.forEach { wordList ->
                    // 因為 Padding 會有重複參考同一個 Bitmap 的狀況，判斷未被 recycle 才進行回收
                    wordList.forEach { bitmap ->
                        if (!bitmap.isRecycled) {
                            bitmap.recycle()
                        }
                    }
                }
                allWordsBitmaps.clear()

            } else {
                withContext(Dispatchers.Main) {
                    updateUICallback?.invoke(listOf("無法辨識：未偵測到嘴部動作"))
                }
            }
        }
    }

    // MediaPipe 臉部偵測與座標計算邏輯
    private fun detectFaceAndGetRect(bitmap: Bitmap): Rect? {
        try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = faceLandmarker.detect(mpImage)

            if (result.faceLandmarks().isEmpty()) return null

            val landmarks = result.faceLandmarks()[0]

            val mouthLeft = landmarks[61]
            val mouthRight = landmarks[291]

            val width = bitmap.width
            val height = bitmap.height

            val leftX = mouthLeft.x() * width
            val leftY = mouthLeft.y() * height
            val rightX = mouthRight.x() * width
            val rightY = mouthRight.y() * height

            val centerX = (leftX + rightX) / 2
            val centerY = (leftY + rightY) / 2

            val mouthWidth = kotlin.math.sqrt(
                (rightX - leftX) * (rightX - leftX) +
                        (rightY - leftY) * (rightY - leftY)
            )

            val cropSize = (mouthWidth * 1.6f).toInt()
            val halfSize = cropSize / 2

            return Rect(
                (centerX - halfSize).toInt(),
                (centerY - halfSize).toInt(),
                (centerX + halfSize).toInt(),
                (centerY + halfSize).toInt()
            )
        } catch (e: Exception) {
            Log.e("LRI", "Face error", e)
        }
        return null
    }

//    private fun toGrayscaleAndResize(bmp: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
//        val scaledBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
//        val canvas = Canvas(scaledBitmap)
//        val paint = Paint()
//        val colorMatrix = ColorMatrix()
//        colorMatrix.setSaturation(0f)
//        val filter = ColorMatrixColorFilter(colorMatrix)
//        paint.colorFilter = filter
//        canvas.drawBitmap(bmp, null, Rect(0, 0, targetWidth, targetHeight), paint)
//        return scaledBitmap
//    }
    private fun toGrayscaleAndResize(bmp: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val enlarge = (targetWidth * 1.1).toInt()      // 96，對齊 dataset.py 的 enlarge_size
        val offset = (enlarge - targetWidth) / 2       // 4

        // 用 PIL 'L' 的亮度係數 (BT.601)，而非 setSaturation(0f) 的 BT.709
        val lum = floatArrayOf(
            0.299f, 0.587f, 0.114f, 0f, 0f,
            0.299f, 0.587f, 0.114f, 0f, 0f,
            0.299f, 0.587f, 0.114f, 0f, 0f,
            0f,     0f,     0f,     1f, 0f
        )

        val big = Bitmap.createBitmap(enlarge, enlarge, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(big)
        val paint = Paint().apply {
            isFilterBitmap = true          // 雙線性，對齊 PIL 的 BILINEAR
            isAntiAlias = true
            colorFilter = ColorMatrixColorFilter(ColorMatrix(lum))
        }
        canvas.drawBitmap(bmp, null, Rect(0, 0, enlarge, enlarge), paint)

        val cropped = Bitmap.createBitmap(big, offset, offset, targetWidth, targetHeight)
        big.recycle()
        return cropped
    }

    // 逐階段延遲統計。所有時間以毫秒輸出，並同時給出每個音節片段的平均值。
    private fun logStageLatency(nSegments: Int, wallClockMs: Long) {
        val ms = 1_000_000.0
        val extract = tExtractNs / ms
        val detect = tDetectNs / ms
        val crop = tCropNs / ms
        val recognize = tRecognizeNs / ms
        val pipeline = extract + detect + crop + recognize
        val per = if (nSegments > 0) nSegments.toDouble() else 1.0

        Log.i("LRI_LATENCY", "=== 逐階段延遲 (音節片段 $nSegments 個, 影格 $nFrames 張) ===")
        Log.i("LRI_LATENCY", "影格抽取        %8.1f ms  (%5.1f ms/音節, %5.2f ms/影格)"
            .format(extract, extract / per, extract / maxOf(nFrames, 1)))
        Log.i("LRI_LATENCY", "唇部偵測        %8.1f ms  (%5.1f ms/音節, %5.2f ms/影格)"
            .format(detect, detect / per, detect / maxOf(nFrames, 1)))
        Log.i("LRI_LATENCY", "裁切/灰階/縮放  %8.1f ms  (%5.1f ms/音節, %5.2f ms/影格)"
            .format(crop, crop / per, crop / maxOf(nFrames, 1)))
        Log.i("LRI_LATENCY", "推論+字典解碼   %8.1f ms  (%5.1f ms/音節)"
            .format(recognize, recognize / per))
        Log.i("LRI_LATENCY", "---------------------------------------------")
        Log.i("LRI_LATENCY", "管線合計        %8.1f ms  (%5.1f ms/音節)".format(pipeline, pipeline / per))
        Log.i("LRI_LATENCY", "錄影結束到結果  %8d ms".format(wallClockMs))
        Log.i("LRI_LATENCY", "存圖除錯: %s".format(if (SAVE_DEBUG_FRAMES) "開啟(數據無效)" else "關閉"))
    }

    private fun saveBitmap(bmp: Bitmap, file: File) {
        try {
            val out = FileOutputStream(file)
            bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
            out.flush()
            out.close()
        } catch (e: Exception) {}
    }

    private fun vibrate() {
        try {
            val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                vibratorManager.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(50, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(50)
            }
        } catch (e: Exception) {}
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val serviceChannel = NotificationChannel(NOTIFICATION_CHANNEL_ID, "LRI Service", NotificationManager.IMPORTANCE_LOW)
            getSystemService(NotificationManager::class.java).createNotificationChannel(serviceChannel)
        }
    }

    private fun buildNotification(): Notification {
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("唇語輸入法")
            .setContentText("服務運行中")
            .setSmallIcon(R.drawable.lri_icon)
            .build()
    }
}