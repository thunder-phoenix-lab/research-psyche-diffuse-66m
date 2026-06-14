package com.psyche.diffuse

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor

/**
 * On-device MDLM (masked diffusion LM) chat engine.
 * fp32 ONNX via ONNX Runtime; tries NNAPI (phone GPU/NPU) with CPU fallback.
 * Generation = confidence-based iterative unmasking (MaskGIT-style), the same
 * decode that fixed repetition collapse on desktop. Emits per-step state for viz.
 */
class MdlmEngine private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val tok: Tokenizer,
    val backend: String,
) {
    companion object {
        const val SEQ_LEN = 512
        const val VOCAB = 16001          // includes [MASK]=16000
        const val MASK_ID = 16000

        fun create(ctx: Context, tok: Tokenizer): MdlmEngine {
            // ORT loads from a file path (mmap, low heap) — copy the asset once.
            val modelFile = File(ctx.filesDir, "mdlm_chat.onnx")
            if (!modelFile.exists() || modelFile.length() == 0L) {
                ctx.assets.open("mdlm_chat.onnx").use { input ->
                    modelFile.outputStream().use { input.copyTo(it, 1 shl 20) }
                }
            }
            val env = OrtEnvironment.getEnvironment()
            // Fastest reliable path on this class of model: multi-threaded ORT CPU (MLAS)
            // + full graph optimization. (NNAPI here ran on its CPU 'reference' backend = no
            // real GPU for these custom diffusion ops, and single-threaded -> slower than this.)
            val cores = Runtime.getRuntime().availableProcessors().coerceIn(2, 8)
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(cores)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val session = env.createSession(modelFile.absolutePath, opts)
            return MdlmEngine(env, session, tok, "CPU x$cores")
        }
    }

    /** Per-step callback: (step, totalSteps, perPositionDisplay) for token visualization. */
    fun interface StepListener {
        fun onStep(step: Int, total: Int, display: List<String>)
    }

    @Synchronized
    fun generate(
        prompt: String,
        respLen: Int = 96,
        steps: Int = 28,
        temperature: Float = 0.9f,
        repPenalty: Float = 1.3f,    // penalize already-committed tokens (kills "It's important to…" loops)
        neighborBan: Int = 2,        // hard-ban token == committed neighbour within ±N (kills "and and")
        noRepeatNgram: Int = 3,      // ban a token that completes a repeated contiguous N-gram
        onStep: StepListener? = null,
    ): String {
        val promptIds = tok.encode(prompt).toMutableList()
        val header = ArrayList<Int>().apply { add(tok.bosId); addAll(promptIds); add(tok.sepId) }
        // truncate prompt from the left if needed
        val maxHeader = SEQ_LEN - respLen - 1
        val hdr = if (header.size > maxHeader) {
            ArrayList<Int>().apply { add(tok.bosId); addAll(header.subList(header.size - maxHeader + 1, header.size)) }
        } else header
        val respStart = hdr.size
        val respEnd = minOf(respStart + respLen, SEQ_LEN)
        val rLen = respEnd - respStart

        val ids = LongArray(SEQ_LEN) { tok.padId.toLong() }
        for (i in hdr.indices) ids[i] = hdr[i].toLong()
        for (i in respStart until respEnd) ids[i] = MASK_ID.toLong()

        val rng = java.util.Random(0)
        fun maskedCount() = (respStart until respEnd).count { ids[it] == MASK_ID.toLong() }

        // reusable lookup tables (avoid per-token hashing): committed tokens (this step) + banned (this pos)
        val committedMask = BooleanArray(VOCAB)
        val bannedMask = BooleanArray(VOCAB)
        val bannedIds = IntArray(64)

        for (step in 0 until steps) {
            val cur = maskedCount()
            if (cur == 0) break
            val tVal = cur.toFloat() / rLen
            val logits = forward(ids, tVal)   // FloatBuffer over [1,512,VOCAB]

            // committed tokens in the response so far -> repetition penalty set (rebuilt each step)
            java.util.Arrays.fill(committedMask, false)
            for (p in respStart until respEnd) {
                val t = ids[p].toInt()
                if (t != MASK_ID && t < VOCAB) committedMask[t] = true
            }

            data class Cand(val pos: Int, val tokId: Int, val conf: Float)
            val cands = ArrayList<Cand>(cur)
            for (pos in respStart until respEnd) {
                if (ids[pos] != MASK_ID.toLong()) continue
                val base = pos * VOCAB

                // --- per-position bans: neighbour (±N) + no-repeat-ngram ---
                var nb = 0
                for (d in 1..neighborBan) {
                    val a = pos - d; val b = pos + d
                    if (a >= respStart && ids[a] != MASK_ID.toLong() && nb < bannedIds.size) bannedIds[nb++] = ids[a].toInt()
                    if (b < respEnd && ids[b] != MASK_ID.toLong() && nb < bannedIds.size) bannedIds[nb++] = ids[b].toInt()
                }
                if (noRepeatNgram >= 3 && pos - 2 >= respStart) {
                    val p1 = ids[pos - 1].toInt(); val p2 = ids[pos - 2].toInt()
                    if (p1 != MASK_ID && p2 != MASK_ID) {  // committed 2-gram (p2,p1); ban whatever followed it elsewhere
                        for (q in respStart until respEnd - 2) {
                            if (ids[q].toInt() == p2 && ids[q + 1].toInt() == p1 &&
                                ids[q + 2] != MASK_ID.toLong() && nb < bannedIds.size
                            ) bannedIds[nb++] = ids[q + 2].toInt()
                        }
                    }
                }
                for (k in 0 until nb) bannedMask[bannedIds[k]] = true

                // --- argmax with repetition penalty + bans ---
                var bestId = -1; var bestLogit = Float.NEGATIVE_INFINITY
                var maxLogit = Float.NEGATIVE_INFINITY
                for (v in 0 until VOCAB - 1) { // exclude MASK_ID (last)
                    var lg = logits.get(base + v)
                    if (lg > maxLogit) maxLogit = lg
                    if (bannedMask[v]) continue
                    if (committedMask[v]) lg = if (lg > 0f) lg / repPenalty else lg * repPenalty
                    if (lg > bestLogit) { bestLogit = lg; bestId = v }
                }
                if (bestId < 0) bestId = 0
                for (k in 0 until nb) bannedMask[bannedIds[k]] = false  // reset for next position

                val conf = 1.0f / (1.0f + kotlin.math.exp(-(bestLogit - maxLogit) / temperature)) +
                        rng.nextFloat() * 1e-3f
                cands.add(Cand(pos, bestId, conf))
            }
            // live prediction for EVERY masked position (preview) — used for the viz so the
            // whole row refreshes together each step, not one token at a time.
            val preview = HashMap<Int, Int>(cands.size * 2)
            for (c in cands) preview[c.pos] = c.tokId

            // cosine schedule: how many should remain masked after this step
            val fracAfter = cos(PI / 2 * (step + 1) / steps).toFloat()
            val targetRemaining = floor(rLen * fracAfter).toInt()
            val nUnmask = maxOf(1, cur - targetRemaining).coerceAtMost(cur)
            cands.sortByDescending { it.conf }
            for (k in 0 until nUnmask) ids[cands[k].pos] = cands[k].tokId.toLong()

            // display: committed token OR live preview for still-masked positions
            // -> every position shows a word and they all update simultaneously, converging
            val row = ArrayList<String>(rLen)
            for (pos in respStart until respEnd) {
                val id = if (ids[pos] != MASK_ID.toLong()) ids[pos].toInt()
                         else (preview[pos] ?: MASK_ID)
                row.add(if (id == MASK_ID) "·" else tok.decode(listOf(id)).ifEmpty { "_" })
            }
            onStep?.onStep(step + 1, steps, row)
        }

        // decode response up to EOS
        val out = ArrayList<Int>(rLen)
        for (pos in respStart until respEnd) {
            val id = ids[pos].toInt()
            if (id == tok.eosId) break
            out.add(id)
        }
        return tok.decode(out)
    }

    private fun forward(ids: LongArray, tVal: Float): FloatBuffer {
        val idsT = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, SEQ_LEN.toLong()))
        val tT = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(tVal)), longArrayOf(1))
        idsT.use { a -> tT.use { b ->
            val res = session.run(mapOf("ids" to a, "t" to b))
            res.use {
                val outT = it.get(0) as OnnxTensor
                // copy out into a heap FloatBuffer so it survives result close
                val src = outT.floatBuffer
                val dst = FloatBuffer.allocate(src.remaining())
                dst.put(src); dst.flip()
                return dst
            }
        } }
    }

    fun close() { session.close() }
}
