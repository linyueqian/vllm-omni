// pcm-worklet-processor — VERBATIM from KoljaB/RealtimeVoiceChat reference.
// Converts Float32 capture frames to Int16 PCM and posts the raw buffer to
// the main thread (transferred). Resampling to 16 kHz mono happens in app.js.
class PCMWorkletProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const in32 = inputs[0][0];
    if (in32) {
      const int16 = new Int16Array(in32.length);
      for (let i = 0; i < in32.length; i++) {
        let s = in32[i]; s = s < -1 ? -1 : s > 1 ? 1 : s;
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);
