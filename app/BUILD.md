# Psyche Diffuse — on-device MDLM chat (experimental APK)

Single-screen chat. The 66.8M clean MDLM (masked diffusion LM) runs fully on-device
via ONNX Runtime. No history, no network. Bonus: live token-unmasking visualization
(you watch the response positions "develop" out of noise — native to diffusion).

## What's here
```
experimental/
  app/src/main/
    assets/  mdlm_chat.onnx (300 MB, fp32, NO quantization)  + vocab.json
    java/com/psyche/diffuse/
      Tokenizer.kt    SP-compatible (greedy longest-match + byte fallback)
      MdlmEngine.kt   ORT session (tries NNAPI = phone GPU/NPU, CPU fallback) + confidence-unmask sampler
      MainActivity.kt Compose chat UI + token-viz
  model/  export_onnx.py (PyTorch->ONNX, parity-checked 100% argmax), finalize_assets.py (vocab dump)
```
Model = `mdlm_sft_clean/final` (clean-corpus 66.8M). fp32, no quant (per request — device has resources;
fp32 is also the best-supported precision for NNAPI/GPU).

## Build (Android Studio — recommended)
1. Open `experimental/` as a project in Android Studio (Giraffe+/latest).
2. On first sync it will: fetch Gradle 8.9, download SDK platform 35, resolve deps
   (Compose BOM, onnxruntime-android 1.20). Accept any SDK download prompts.
3. Plug in a phone (USB debugging) and Run. First launch copies the 300 MB model to
   app storage (a few seconds) then shows `ready · NNAPI(GPU/NPU)` or `ready · CPU`.

## Build (CLI, if you prefer)
From `experimental/` after `gradle wrapper` has materialized `gradlew`:
```
gradlew.bat assembleDebug
# APK -> app/build/outputs/apk/debug/app-debug.apk
```
(JDK is bundled with Android Studio at `C:\Program Files\Android\Android Studio\jbr`.)

## Notes / honest caveats
- **Untested on a device from the dev environment** — built blind; expect to fix 1-2 sync/version
  nits in Android Studio (it auto-suggests AGP/Gradle bumps).
- **Speed:** ~28 diffusion steps × one 66.8M forward each. On NNAPI (GPU/NPU) expect a few seconds;
  pure CPU is slower. The token-viz makes the wait legible.
- **Quality:** this is a 66.8M model — coherent psychology-flavored sentences, not factual QA.
  Coping-style prompts ("I feel anxious about work") work best.
- NNAPI op-coverage varies by phone; ORT silently falls back to CPU per-op if needed.
