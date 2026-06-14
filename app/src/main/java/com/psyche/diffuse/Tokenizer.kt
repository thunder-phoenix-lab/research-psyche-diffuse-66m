package com.psyche.diffuse

import android.content.Context
import org.json.JSONObject

/**
 * Minimal SentencePiece-compatible tokenizer (no native deps).
 * - encode: SP normalization (space -> U+2581 '▁', leading '▁') then greedy longest-match
 *   over the vocab, with byte fallback (<0xXX>) for anything unmatched.
 * - decode: pieces -> text, reassemble <0xXX> byte runs, '▁' -> space.
 * Greedy longest-match is not byte-identical to SP-BPE but is close and robust for a demo.
 */
class Tokenizer private constructor(
    private val pieces: List<String>,
    val unkId: Int, val bosId: Int, val eosId: Int, val padId: Int,
    val sepId: Int, val maskId: Int,
) {
    private val pieceToId = HashMap<String, Int>(pieces.size * 2).apply {
        pieces.forEachIndexed { i, p -> put(p, i) }
    }
    private val maxPieceLen = pieces.maxOf { it.length }
    private val byteRe = Regex("^<0x([0-9A-Fa-f]{2})>$")

    companion object {
        const val SPACE = '▁'  // ▁
        fun fromAssets(ctx: Context, name: String = "vocab.json"): Tokenizer {
            val txt = ctx.assets.open(name).bufferedReader().use { it.readText() }
            val o = JSONObject(txt)
            val arr = o.getJSONArray("pieces")
            val ps = ArrayList<String>(arr.length())
            for (i in 0 until arr.length()) ps.add(arr.getString(i))
            return Tokenizer(
                ps,
                unkId = o.getInt("unk_id"), bosId = o.getInt("bos_id"),
                eosId = o.getInt("eos_id"), padId = o.getInt("pad_id"),
                sepId = o.getInt("sep_id"), maskId = o.getInt("mask_id"),
            )
        }
    }

    fun encode(text: String): IntArray {
        val norm = (SPACE + text.trim().replace(' ', SPACE))
        val out = ArrayList<Int>(norm.length)
        var i = 0
        while (i < norm.length) {
            var matched = false
            val maxLen = minOf(maxPieceLen, norm.length - i)
            var l = maxLen
            while (l >= 1) {
                val sub = norm.substring(i, i + l)
                val id = pieceToId[sub]
                if (id != null) { out.add(id); i += l; matched = true; break }
                l--
            }
            if (!matched) {
                // byte fallback for this single codepoint
                val cp = norm.codePointAt(i)
                val s = String(Character.toChars(cp))
                for (b in s.toByteArray(Charsets.UTF_8)) {
                    val hex = "<0x%02X>".format(b.toInt() and 0xFF)
                    out.add(pieceToId[hex] ?: unkId)
                }
                i += s.length
            }
        }
        return out.toIntArray()
    }

    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        val pending = ArrayList<Byte>()
        fun flushBytes() {
            if (pending.isNotEmpty()) {
                sb.append(String(pending.toByteArray(), Charsets.UTF_8))
                pending.clear()
            }
        }
        for (id in ids) {
            if (id == padId || id == bosId || id == eosId || id == sepId || id == maskId) continue
            if (id < 0 || id >= pieces.size) continue
            val p = pieces[id]
            val m = byteRe.find(p)
            if (m != null) {
                pending.add(m.groupValues[1].toInt(16).toByte())
            } else {
                flushBytes()
                sb.append(p)
            }
        }
        flushBytes()
        return sb.toString().replace(SPACE, ' ').trim()
    }
}
