package com.example.lri

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class LipReadingManager(private val context: Context) {

    private val isMockMode = false

    private var tfliteInterpreter: Interpreter? = null
    private val lipProcessor = LipReadingProcessor(context)
    private val smartConverter = SmartPinyinConverter(context)

    private val INPUT_SIZE = 88
    private val NUM_FRAMES = 40
    private val NUM_CLASSES = 187
    private val MODEL_NAME = "final_model_float32.tflite"

    // 加入狀態標記，防止模型還沒載入完就被呼叫
    var isModelReady = false
        private set

    init {
        if (!isMockMode) {
            // 解決 Input Latency：將載入模型的沉重 I/O 操作移至背景執行
            CoroutineScope(Dispatchers.IO).launch {
                loadModel()
            }
        }
    }

    private fun loadModel() {
        try {
            val assetFileDescriptor = context.assets.openFd(MODEL_NAME)
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            tfliteInterpreter = Interpreter(mappedByteBuffer, options)

            tfliteInterpreter?.let { itp ->
                val inT = itp.getInputTensor(0)
                val outT = itp.getOutputTensor(0)
                Log.d("LRI_DEBUG", "INPUT  shape = [${inT.shape().joinToString()}]  dtype = ${inT.dataType()}")
                Log.d("LRI_DEBUG", "OUTPUT shape = [${outT.shape().joinToString()}]  dtype = ${outT.dataType()}")
            }

            isModelReady = true
            Log.d("LRI_DEBUG", "✅ TFLite 模型載入成功 (背景)")
            benchmarkInference()
        } catch (e: Exception) {
            Log.e("LRI_DEBUG", "❌ TFLite 模型載入失敗", e)
        }
    }

    // 解決 Lifecycle Stability：加入安全釋放資源的方法
    fun close() {
        try {
            tfliteInterpreter?.close()
            tfliteInterpreter = null
            isModelReady = false
            Log.d("LRI_DEBUG", "✅ TFLite 模型已安全釋放")
        } catch (e: Exception) {
            Log.e("LRI_DEBUG", "釋放模型時發生錯誤", e)
        }
    }

    fun recognizeSentence(wordBitmaps: List<List<Bitmap>>): List<String> {
        if (!isModelReady) return listOf("系統準備中，請稍後...")

        val pinyinSequenceCandidates = ArrayList<List<String>>()

        // ---- 逐階段延遲量測 ----
        var tPackNs = 0L        // 張量打包
        var tInferNs = 0L       // TFLite 推論
        var tDecodeNs = 0L      // 字典解碼

        for ((index, frames) in wordBitmaps.withIndex()) {
            val tPack0 = System.nanoTime()
            val inputBuffer = preprocessBitmaps(frames)
            tPackNs += System.nanoTime() - tPack0

            if (inputBuffer != null) {
                var topPinyins = listOf("unknown")

                if (!isMockMode && tfliteInterpreter != null) {
                    try {
                        val outputArray = Array(1) { FloatArray(NUM_CLASSES) }
                        val tInfer0 = System.nanoTime()
                        tfliteInterpreter?.run(inputBuffer, outputArray)
                        tInferNs += System.nanoTime() - tInfer0

                        topPinyins = lipProcessor.getTopKPinyins(outputArray[0], 3)
                        lipProcessor.decodeAndLog(outputArray[0], index + 1)
                    } catch (e: Exception) {
                        Log.e("LRI_DEBUG", "推論崩潰", e)
                    }
                }
                if (topPinyins.isNotEmpty() && topPinyins[0] != "unknown") {
                    pinyinSequenceCandidates.add(topPinyins)
                }
            }
        }

        if (pinyinSequenceCandidates.isEmpty()) {
            logRecognizeLatency(wordBitmaps.size, tPackNs, tInferNs, 0L)
            return listOf("無法辨識拼音")
        }

        val tDecode0 = System.nanoTime()
        val sentences = smartConverter.convertToSentence(pinyinSequenceCandidates)
        tDecodeNs = System.nanoTime() - tDecode0

        logRecognizeLatency(wordBitmaps.size, tPackNs, tInferNs, tDecodeNs)
        return sentences
    }

    private fun logRecognizeLatency(nSegments: Int, packNs: Long, inferNs: Long, decodeNs: Long) {
        val ms = 1_000_000.0
        val per = if (nSegments > 0) nSegments.toDouble() else 1.0
        val pack = packNs / ms
        val infer = inferNs / ms
        val decode = decodeNs / ms
        Log.i("LRI_LATENCY", "張量打包        %8.1f ms  (%5.1f ms/音節)".format(pack, pack / per))
        Log.i("LRI_LATENCY", "TFLite 推論     %8.1f ms  (%5.1f ms/音節)".format(infer, infer / per))
        Log.i("LRI_LATENCY", "字典解碼        %8.1f ms  (整句一次)".format(decode))
    }

    private fun preprocessBitmaps(frames: List<Bitmap>): ByteBuffer? {
        val bufferSize = 1 * NUM_FRAMES * INPUT_SIZE * INPUT_SIZE * 1 * 4
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        // 保持你原本正確的記憶體排列方式
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                for (i in 0 until NUM_FRAMES) {
                    var r = 0
                    if (i < frames.size) {
                        val bitmap = frames[i]
                        if (x < bitmap.width && y < bitmap.height) {
                            val pixel = bitmap.getPixel(x, y)
                            r = (pixel shr 16) and 0xFF
                        }
                    }
                    // TODO: 這裡請換回你確認過的正規化公式 (除以 255 或是 -0.5 / 0.5 等)
                    byteBuffer.putFloat(r / 255.0f)
                }
            }
        }
        byteBuffer.rewind()
        return byteBuffer
    }

    fun benchmarkInference(warmup: Int = 10, iters: Int = 50) {
        val interpreter = tfliteInterpreter
        if (interpreter == null || !isModelReady) {
            Log.e("LRI_BENCH", "模型尚未載入完成，無法測試")
            return
        }

        // 1. 假輸入：大小跟真實前處理完全一樣（延遲跟數值無關，填 0 即可）
        val bufferSize = 1 * NUM_FRAMES * INPUT_SIZE * INPUT_SIZE * 1 * 4
        val input = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder())
        val output = Array(1) { FloatArray(NUM_CLASSES) }

        // 2. Warm-up：不計時，甩掉冷啟動
        repeat(warmup) {
            input.rewind()
            interpreter.run(input, output)
        }

        // 3. 正式計時
        val times = DoubleArray(iters)
        repeat(iters) { i ->
            input.rewind()
            val t0 = System.nanoTime()
            interpreter.run(input, output)
            val t1 = System.nanoTime()
            times[i] = (t1 - t0) / 1_000_000.0   // ns → ms
        }

        // 4. 統計
        val mean = times.average()
        val std = kotlin.math.sqrt(times.map { (it - mean) * (it - mean) }.average())
        val sorted = times.sortedArray()
        val median = sorted[iters / 2]
        val p95 = sorted[(iters * 95 / 100).coerceAtMost(iters - 1)]


        Log.i("LRI_BENCH", "=== 純模型推論延遲 (n=$iters, threads=4) ===")
        Log.i("LRI_BENCH", "Mean=%.2f ms  Std=%.2f ms".format(mean, std))
        Log.i("LRI_BENCH", "Median=%.2f  P95=%.2f  Min=%.2f  Max=%.2f".format(median, p95, sorted.first(), sorted.last()))
    }
}

