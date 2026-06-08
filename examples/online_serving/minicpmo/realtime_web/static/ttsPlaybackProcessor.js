// tts-playback-processor — VERBATIM from KoljaB/RealtimeVoiceChat reference.
// Gapless FIFO player for Int16Array chunks. Underrun -> silence (graceful).
// postMessage({type:'clear'}) flushes the queue for barge-in.
class TTSPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferQueue = [];
    this.readOffset = 0;
    this.samplesRemaining = 0;
    this.isPlaying = false;
    this.port.onmessage = (e) => {
      if (e.data && typeof e.data === 'object' && e.data.type === 'clear') {
        this.bufferQueue = [];
        this.readOffset = 0;
        this.samplesRemaining = 0;
        this.isPlaying = false;
        return;
      }
      this.bufferQueue.push(e.data);
      this.samplesRemaining += e.data.length;
    };
  }
  process(inputs, outputs) {
    const out = outputs[0][0];
    if (this.samplesRemaining === 0) {
      out.fill(0);
      if (this.isPlaying) {
        this.isPlaying = false;
        this.port.postMessage({ type: 'ttsPlaybackStopped' });
      }
      return true;
    }
    if (!this.isPlaying) {
      this.isPlaying = true;
      this.port.postMessage({ type: 'ttsPlaybackStarted' });
    }
    let i = 0;
    while (i < out.length && this.bufferQueue.length > 0) {
      const b = this.bufferQueue[0];
      out[i++] = b[this.readOffset] / 32768;
      this.readOffset++;
      this.samplesRemaining--;
      if (this.readOffset >= b.length) {
        this.bufferQueue.shift();
        this.readOffset = 0;
      }
    }
    while (i < out.length) out[i++] = 0;
    return true;
  }
}
registerProcessor('tts-playback-processor', TTSPlaybackProcessor);
