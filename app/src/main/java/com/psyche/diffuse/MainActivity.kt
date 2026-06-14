package com.psyche.diffuse

import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { ChatApp(applicationContext) } }
    }
}

@Composable
fun ChatApp(ctx: Context) {
    val main = remember { Handler(Looper.getMainLooper()) }
    var engine by remember { mutableStateOf<MdlmEngine?>(null) }
    var status by remember { mutableStateOf("loading model…") }

    LaunchedEffect(Unit) {
        withContext(Dispatchers.Default) {
            val tok = Tokenizer.fromAssets(ctx)
            val e = MdlmEngine.create(ctx, tok)
            main.post { engine = e; status = "ready · ${e.backend}" }
        }
    }

    val messages = remember { mutableStateListOf<Msg>() }
    var input by remember { mutableStateOf("") }
    var generating by remember { mutableStateOf(false) }
    var viz by remember { mutableStateOf<List<String>>(emptyList()) }
    var step by remember { mutableStateOf("") }
    val scroll = rememberScrollState()
    val scope = rememberCoroutineScope()

    // tunable model params
    var showParams by remember { mutableStateOf(false) }
    var pSteps by remember { mutableStateOf(28f) }
    var pTemp by remember { mutableStateOf(0.9f) }
    var pRep by remember { mutableStateOf(1.3f) }
    var pRespLen by remember { mutableStateOf(96f) }

    fun send() {
        val e = engine ?: return
        val p = input.trim()
        if (p.isEmpty() || generating) return
        messages.add(Msg(true, p))
        input = ""; generating = true; viz = emptyList(); step = ""
        scope.launch {
            val resp = withContext(Dispatchers.Default) {
                e.generate(
                    "user: $p assistant:",
                    respLen = pRespLen.toInt(), steps = pSteps.toInt(),
                    temperature = pTemp, repPenalty = pRep,
                ) { s, t, disp -> main.post { viz = disp; step = "diffusion step $s/$t" } }
            }
            messages.add(Msg(false, resp.ifBlank { "…" }))
            generating = false; viz = emptyList(); step = ""
        }
    }

    LaunchedEffect(messages.size, viz) { scroll.animateScrollTo(scroll.maxValue) }

    Column(Modifier.fillMaxSize().background(Color(0xFF101418)).safeDrawingPadding().padding(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Psyche Diffuse", color = Color(0xFFE6EDF3), fontWeight = FontWeight.Bold, fontSize = 18.sp)
                Text(status, color = Color(0xFF7D8590), fontSize = 12.sp)
            }
            TextButton(onClick = { showParams = !showParams }) {
                Text(if (showParams) "hide" else "params", color = Color(0xFF58A6FF), fontSize = 13.sp)
            }
        }
        if (showParams) {
            ParamRow("steps", pSteps, 8f..48f, 40) { pSteps = it }
            ParamRow("temp", pTemp, 0.4f..1.4f, 0) { pTemp = it }
            ParamRow("rep-pen", pRep, 1.0f..1.8f, 0) { pRep = it }
            ParamRow("len", pRespLen, 32f..160f, 128) { pRespLen = it }
        }
        Spacer(Modifier.height(8.dp))

        Column(Modifier.weight(1f).fillMaxWidth().verticalScroll(scroll)) {
            messages.forEach { Bubble(it) }
            if (generating) VizPanel(step, viz)
        }

        Spacer(Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier.weight(1f).background(Color(0xFF1C2128), RoundedCornerShape(10.dp)).padding(12.dp)
            ) {
                BasicTextField(
                    value = input, onValueChange = { input = it },
                    textStyle = androidx.compose.ui.text.TextStyle(color = Color(0xFFE6EDF3), fontSize = 15.sp),
                    cursorBrush = androidx.compose.ui.graphics.SolidColor(Color(0xFF58A6FF)),
                    modifier = Modifier.fillMaxWidth(),
                    decorationBox = { inner ->
                        if (input.isEmpty()) Text("Ask something…", color = Color(0xFF545D68), fontSize = 15.sp)
                        inner()
                    },
                )
            }
            Spacer(Modifier.width(8.dp))
            Button(
                onClick = { send() },
                enabled = engine != null && !generating,
                shape = RoundedCornerShape(10.dp),
            ) { Text(if (generating) "…" else "Send") }
        }
    }
}

data class Msg(val user: Boolean, val text: String)

@Composable
fun Bubble(m: Msg) {
    val bg = if (m.user) Color(0xFF1F6FEB) else Color(0xFF21262D)
    val align = if (m.user) Alignment.End else Alignment.Start
    Column(Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalAlignment = align) {
        Box(
            Modifier.widthIn(max = 320.dp).background(bg, RoundedCornerShape(12.dp)).padding(10.dp)
        ) { Text(m.text, color = Color(0xFFE6EDF3), fontSize = 15.sp) }
    }
}

@Composable
fun ParamRow(
    label: String, value: Float,
    range: ClosedFloatingPointRange<Float>, sliderSteps: Int,
    onChange: (Float) -> Unit,
) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().height(30.dp)) {
        Text(label, color = Color(0xFF7D8590), fontSize = 12.sp, modifier = Modifier.width(58.dp))
        val disp = if (sliderSteps > 0) value.toInt().toString() else String.format("%.1f", value)
        Text(disp, color = Color(0xFFE6EDF3), fontSize = 12.sp, modifier = Modifier.width(30.dp))
        Slider(
            value = value, onValueChange = onChange, valueRange = range, steps = sliderSteps,
            modifier = Modifier.weight(1f),
        )
    }
}

/** Live diffusion token grid: '·' = still masked, text = unmasked piece. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun VizPanel(step: String, tokens: List<String>) {
    Column(Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        Text(step, color = Color(0xFF58A6FF), fontSize = 11.sp)
        Spacer(Modifier.height(4.dp))
        FlowRow(Modifier.fillMaxWidth()) {
            tokens.forEach { t ->
                val masked = t == "·"
                Box(
                    Modifier.padding(1.dp)
                        .background(if (masked) Color(0xFF161B22) else Color(0xFF2D333B), RoundedCornerShape(3.dp))
                        .padding(horizontal = 3.dp, vertical = 1.dp)
                ) {
                    Text(
                        if (masked) "·" else t,
                        color = if (masked) Color(0xFF3D444D) else Color(0xFF7EE787),
                        fontSize = 11.sp, fontFamily = FontFamily.Monospace,
                    )
                }
            }
        }
    }
}
