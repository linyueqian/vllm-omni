/**
 * lib/realtime-duplex-session.js — DuplexSession drop-in over vLLM realtime.
 *
 * Same public surface as duplex-session.js (hooks, sendChunk, pause,
 * force-listen, AudioPlayer coordination) but speaks the OpenAI-style
 * realtime duplex protocol of vLLM-Omni's ``/v1/realtime?duplex=1``
 * directly from the browser — no gateway, no worker, no protocol bridge.
 * The page's audio pipeline is unchanged: mic chunks arrive here as
 * f32le 16 kHz base64 and reply audio is handed to AudioPlayer as
 * f32le 24 kHz base64.
 */

import { AudioPlayer } from './audio-player.js';
import { resampleAudio } from './duplex-utils.js';

const INPUT_RATE = 16000;
const SILENCE_COMMIT_MS = 500;
const SPEECH_RMS = 0.015;
const CHUNK_MS_EST = 1000;
// If the server has not started a response this long after a commit,
// send response.create (the auto-response only fires on the first turn).
const RESPONSE_CREATE_FALLBACK_MS = 900;

// ---- base64 <-> typed array helpers -------------------------------------

function b64ToBytes(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
}

function bytesToB64(bytes) {
    let binary = '';
    for (let i = 0; i < bytes.length; i += 8192) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
    }
    return btoa(binary);
}

/** f32le base64 (mic chunk) -> pcm16 base64 (realtime append payload). */
function f32B64ToPcm16B64(b64) {
    const f32 = new Float32Array(b64ToBytes(b64).buffer);
    const pcm = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        pcm[i] = s < 0 ? s * 32768 : s * 32767;
    }
    return bytesToB64(new Uint8Array(pcm.buffer));
}

/** Decode one realtime audio delta to Float32Array at 24 kHz. */
function deltaToF32(evt, outputRate) {
    const encoded = evt.delta || (evt.response && evt.response.audio) || '';
    if (!encoded) return null;
    const bytes = b64ToBytes(encoded);
    const fmt = String(evt.format || evt.audio_format || 'pcm16').toLowerCase();
    let f32;
    if (fmt.includes('f32')) {
        f32 = new Float32Array(bytes.buffer, 0, Math.floor(bytes.byteLength / 4));
    } else {
        const pcm = new Int16Array(bytes.buffer, 0, Math.floor(bytes.byteLength / 2));
        f32 = new Float32Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768;
    }
    const rate = Number(evt.sample_rate_hz || evt.sample_rate || outputRate);
    if (rate !== outputRate && f32.length) f32 = resampleAudio(f32, rate, outputRate);
    return f32;
}

/** f32le 16 kHz base64 -> WAV data URI (for extra_body.ref_audio). */
function f32B64ToWavDataUri(b64, sampleRate = INPUT_RATE) {
    const f32 = new Float32Array(b64ToBytes(b64).buffer);
    const pcm = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        pcm[i] = s < 0 ? s * 32768 : s * 32767;
    }
    const dataLen = pcm.length * 2;
    const buf = new ArrayBuffer(44 + dataLen);
    const view = new DataView(buf);
    const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    writeStr(0, 'RIFF'); view.setUint32(4, 36 + dataLen, true); writeStr(8, 'WAVE');
    writeStr(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    writeStr(36, 'data'); view.setUint32(40, dataLen, true);
    new Uint8Array(buf, 44).set(new Uint8Array(pcm.buffer));
    return 'data:audio/wav;base64,' + bytesToB64(new Uint8Array(buf));
}

// ---- session -------------------------------------------------------------

export class DuplexSession {
    constructor(prefix, config = {}) {
        this.prefix = prefix;
        this.config = {
            getMaxKvTokens: config.getMaxKvTokens || (() => 8192),
            getPlaybackDelayMs: config.getPlaybackDelayMs || (() => 200),
            outputSampleRate: config.outputSampleRate || 24000,
            getWsUrl: config.getWsUrl || (() => {
                const proto = location.protocol === 'https:' ? 'wss' : 'ws';
                return `${proto}://${location.host}/v1/realtime?duplex=1`;
            }),
        };

        this.ws = null;
        this.audioPlayer = new AudioPlayer({
            outputSampleRate: this.config.outputSampleRate,
            getPlaybackDelayMs: this.config.getPlaybackDelayMs,
        });
        this.sessionId = '';
        this.chunksSent = 0;
        this._hadSpeech = false;
        this._silenceMs = 0;
        this.paused = false;
        this.pauseState = 'active';
        this.forceListenActive = false;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this._started = false;
        this._lastSpeechStopTime = 0;
        this._firstDeltaOfResponse = true;
        this._responseCreateTimer = null;
        this._responseActive = false;

        this.audioPlayer.onMetrics = (data) => {
            this.onMetrics({
                type: 'audio',
                ahead: data.ahead,
                gapCount: data.gapCount,
                totalShift: data.totalShift,
                turn: data.turn,
                pdelay: data.pdelay,
            });
        };
    }

    get running() { return this._started; }

    // ==== Hooks — identical surface to duplex-session.js ====
    onSystemLog(text) {}
    onQueueUpdate(data) {}
    onQueueDone() {}
    onSpeakStart(text) { return null; }
    onSpeakUpdate(handle, text) {}
    onSpeakEnd() {}
    onListenResult(result) {}
    onExtraResult(result, recvTime) {}
    async onPrepared() {}
    onCleanup() {}
    onMetrics(data) {}
    onRunningChange(running) {}
    onPauseStateChange(state) {}
    onForceListenChange(active) {}

    // ==== Core API ====

    async start(systemPrompt, preparePayload, startMediaFn) {
        this._reset();
        this.sessionId = `${this.prefix}_${Date.now().toString(36)}`;
        this.onMetrics({ type: 'state', sessionState: 'Connecting...', sessionId: this.sessionId });

        const wsUrl = this.config.getWsUrl(this.sessionId);

        try {
            await new Promise((resolve, reject) => {
                this.ws = new WebSocket(wsUrl);
                // The omni page writes client_diagnostic events straight to
                // session.ws; the realtime endpoint rejects unknown events,
                // so drop them at the socket.
                const rawSend = this.ws.send.bind(this.ws);
                this.ws.send = (data) => {
                    if (typeof data === 'string' && data.includes('"client_diagnostic"')) return;
                    rawSend(data);
                };
                this.ws.onopen = () => resolve();
                this.ws.onerror = () => reject(new Error('WebSocket connection failed'));
                this.ws.onclose = () => {
                    if (!this._started) reject(new Error('WebSocket closed before ready'));
                };
            });

            // Realtime session negotiation replaces the worker 'prepare'.
            await new Promise((resolve, reject) => {
                const extraBody = {
                    auto_response: true,
                    minicpmo45_native_duplex: true,
                };
                const refB64 = preparePayload && (
                    preparePayload.tts_ref_audio_base64 || preparePayload.ref_audio_base64);
                if (refB64) extraBody.ref_audio = f32B64ToWavDataUri(refB64);

                this.ws.onmessage = (e) => {
                    let msg;
                    try { msg = JSON.parse(e.data); } catch (_) { return; }
                    if (msg.type === 'session.created' || msg.type === 'session.updated') {
                        this.onQueueUpdate(null);
                        this.onSystemLog('Prepared (vLLM realtime session)');
                        resolve();
                    } else if (msg.type === 'error') {
                        reject(new Error(JSON.stringify(msg.error || msg)));
                    }
                };

                this.ws.send(JSON.stringify({
                    type: 'session.update',
                    session: {
                        modalities: ['audio', 'text'],
                        instructions: systemPrompt || '',
                        input_audio_format: 'pcm16',
                        output_audio_format: 'pcm16',
                        extra_body: extraBody,
                    },
                }));

                setTimeout(() => reject(new Error('session negotiation timed out')), 30000);
            });

            await this.onPrepared();
            this.audioPlayer.init();
            if (startMediaFn) await startMediaFn();

            this._started = true;
            this.onRunningChange(true);
            this.ws.onmessage = (e) => {
                let msg;
                try { msg = JSON.parse(e.data); } catch (_) { return; }
                this._handleEvent(msg);
            };
            this.ws.onclose = () => {
                this.onSystemLog('Session ended');
                this.cleanup();
            };
        } catch (err) {
            if (this.ws) { try { this.ws.close(); } catch (_) {} this.ws = null; }
            this._started = false;
            throw err;
        }
    }

    /**
     * Page sends {type:'audio_chunk', audio_base64:<f32le16k b64>,
     * frame_base64_list?, max_slice_nums?} — translate to a realtime append.
     * The omni page carries camera frames; a per-utterance commit is injected
     * because the runtime schedules a response on input_audio_buffer.commit.
     */
    sendChunk(msg) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this.paused) return;
        const event = {
            type: 'input_audio_buffer.append',
            audio: f32B64ToPcm16B64(msg.audio_base64),
            format: 'pcm16',
            sample_rate_hz: INPUT_RATE,
        };
        if (this.forceListenActive) event.force_listen = true;
        // Omni page camera frames (base64 JPEG). HD slicing is unsupported by
        // the duplex adapter, so only max_slice_nums<=1 frames are forwarded.
        if (Array.isArray(msg.frame_base64_list) && msg.frame_base64_list.length
            && (!msg.max_slice_nums || msg.max_slice_nums <= 1)) {
            event.video_frames = msg.frame_base64_list;
        }
        this.ws.send(JSON.stringify(event));
        this.chunksSent++;
        this.onMetrics({ type: 'result', chunksSent: this.chunksSent });

        // End-of-utterance commit (client VAD): the page never commits, but the
        // runtime creates a response on commit.
        let sumSq = 0;
        const f32 = new Float32Array(b64ToBytes(msg.audio_base64).buffer);
        for (let i = 0; i < f32.length; i += 1) sumSq += f32[i] * f32[i];
        const rms = f32.length ? Math.sqrt(sumSq / f32.length) : 0;
        const durMs = f32.length ? Math.round((f32.length / INPUT_RATE) * 1000) : CHUNK_MS_EST;
        if (rms > SPEECH_RMS) {
            this._hadSpeech = true;
            this._silenceMs = 0;
        } else if (this._hadSpeech) {
            this._silenceMs += durMs;
            if (this._silenceMs >= SILENCE_COMMIT_MS) {
                this._hadSpeech = false;
                this._silenceMs = 0;
                this.ws.send(JSON.stringify({ type: 'input_audio_buffer.commit', final: true }));
                // The runtime auto-responds to the first commit but not to
                // later ones (post-response commits are mis-deferred as
                // barge-in of an already-finished response). If no response
                // starts shortly, request one explicitly — the runtime then
                // replays the committed audio.
                if (this._responseCreateTimer) clearTimeout(this._responseCreateTimer);
                this._responseCreateTimer = setTimeout(() => {
                    this._responseCreateTimer = null;
                    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || this.paused) return;
                    // Mid-response (barge-in) commits are deferred server-side
                    // and start automatically when the active response ends.
                    if (this._responseActive) return;
                    this.ws.send(JSON.stringify({ type: 'response.create' }));
                }, RESPONSE_CREATE_FALLBACK_MS);
            }
        }
    }

    /** Client-side pause: stop feeding audio (no server pause in realtime). */
    pauseToggle() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (this.pauseState === 'active') {
            this.paused = true;
            this.pauseState = 'paused';
            if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
            this.onPauseStateChange('paused');
            this.onMetrics({ type: 'state', sessionState: 'Paused' });
            this.onSystemLog('Session paused (input muted)');
        } else if (this.pauseState === 'paused') {
            this.paused = false;
            this.pauseState = 'active';
            this.onPauseStateChange('active');
            this.onMetrics({ type: 'state', sessionState: 'Active' });
            this.onSystemLog('Session resumed');
        }
    }

    toggleForceListen() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        this.forceListenActive = !this.forceListenActive;
        this.onForceListenChange(this.forceListenActive);
        if (this.forceListenActive) {
            this.onSystemLog('Force Listen ON — model will only listen');
            this.audioPlayer.stopAll();
            if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
        } else {
            if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
            this.onSystemLog('Force Listen OFF — model may speak');
        }
    }

    stop() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try { this.ws.send(JSON.stringify({ type: 'session.close' })); } catch (_) {}
        }
        this.cleanup();
    }

    cancelQueue() { this.cleanup(); }

    cleanup() {
        this.onCleanup();
        if (this._responseCreateTimer) {
            clearTimeout(this._responseCreateTimer);
            this._responseCreateTimer = null;
        }
        this._responseActive = false;
        this.audioPlayer.stop();
        if (this.ws) {
            this.ws.onclose = null;
            try { this.ws.close(); } catch (_) {}
            this.ws = null;
        }
        this._started = false;
        this.paused = false;
        this.pauseState = 'active';
        this.forceListenActive = false;
        this.onRunningChange(false);
        this.onForceListenChange(false);
        this.onPauseStateChange('active');
        this.onMetrics({ type: 'state', sessionState: 'Stopped' });
    }

    // ==== Internal ====

    _reset() {
        this.chunksSent = 0;
        this._hadSpeech = false;
        this._silenceMs = 0;
        this.currentSpeakText = '';
        this._speakHandle = null;
        this.paused = false;
        this.pauseState = 'active';
        this.forceListenActive = false;
        this._lastSpeechStopTime = 0;
        this._firstDeltaOfResponse = true;
    }

    _emitModelState(state, extra = {}) {
        this.onMetrics({
            type: 'result',
            modelState: state,
            chunksSent: this.chunksSent,
            ...extra,
        });
    }

    _handleEvent(evt) {
        const now = performance.now();
        switch (evt.type) {
            case 'response.created':
                this._responseActive = true;
                if (this._responseCreateTimer) {
                    clearTimeout(this._responseCreateTimer);
                    this._responseCreateTimer = null;
                }
                break;

            case 'response.audio.delta':
            case 'response.output_audio.delta': {
                const f32 = deltaToF32(evt, this.config.outputSampleRate);
                if (f32 && f32.length) {
                    if (!this.audioPlayer.turnActive) this.audioPlayer.beginTurn();
                    this.audioPlayer.playChunk(bytesToB64(new Uint8Array(f32.buffer)), now);
                }
                if (this._firstDeltaOfResponse) {
                    this._firstDeltaOfResponse = false;
                    const ttfs = this._lastSpeechStopTime > 0 ? now - this._lastSpeechStopTime : null;
                    this._emitModelState('speaking', ttfs ? { ttfsMs: ttfs } : {});
                } else {
                    this._emitModelState('speaking');
                }
                break;
            }

            // Transcript deltas are the only response-text channel.
            case 'response.audio_transcript.delta':
            case 'response.output_audio_transcript.delta': {
                const delta = evt.delta;
                if (typeof delta === 'string' && delta) {
                    this.currentSpeakText += delta;
                    if (!this._speakHandle) {
                        this._speakHandle = this.onSpeakStart(this.currentSpeakText);
                    } else {
                        this.onSpeakUpdate(this._speakHandle, this.currentSpeakText);
                    }
                }
                break;
            }

            case 'response.listen':
                if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
                this._emitModelState('listening');
                break;

            case 'input_audio_buffer.speech_started':
                this._emitModelState('listening');
                break;

            case 'input_audio_buffer.speech_stopped':
                this._lastSpeechStopTime = now;
                break;

            case 'conversation.item.input_audio_transcription.completed': {
                const t = evt.transcript;
                if (typeof t === 'string' && t.trim()) this.onListenResult({ text: t, is_listen: true });
                break;
            }

            case 'response.done': {
                this._responseActive = false;
                if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
                if (this._speakHandle) this.onSpeakEnd();
                this._speakHandle = null;
                this.currentSpeakText = '';
                this._firstDeltaOfResponse = true;
                this.onSystemLog('— end of turn —');
                this._emitModelState('end_of_turn');
                break;
            }

            case 'response.cancelled':
            case 'output_audio_buffer.clear':
            case 'output_audio_buffer.cleared':
                this._responseActive = false;
                this.audioPlayer.stopAll();
                if (this.audioPlayer.turnActive) this.audioPlayer.endTurn();
                this._speakHandle = null;
                this.currentSpeakText = '';
                this._firstDeltaOfResponse = true;
                this._emitModelState('listening');
                break;

            case 'error': {
                const detail = evt.error ? JSON.stringify(evt.error) : JSON.stringify(evt);
                this.onSystemLog(`Error: ${detail}`);
                break;
            }

            default:
                break;
        }
        this.onExtraResult(evt, now);
    }
}
