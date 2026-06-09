// Realtime duplex voice chat for MiniCPM-o 4.5.
// mic (AudioWorklet -> 16k mono pcm16) --> WS /v1/realtime?duplex=1 --> 24k playback (AudioWorklet).
//
// Interaction modes (client-side behaviors over the same WS):
//   full : continuous mic; model-driven turn-taking (or server_vad). Current default behavior.
//   half : push-to-talk; mic frames are only sent while "Hold to talk" is held.
//   turn : record one utterance (hold), then on release send commit{final}+response.create.
//
// Voice: chosen BEFORE the call in the initial session.update.
//   named voice -> session.voice (e.g. "default")
//   reference-audio clone -> session.extra_body.ref_audio = data:audio/wav;base64,...
//   (neither can change after audio output starts -> server rejects voice_update_after_audio).
//
// Barge-in: a new listen/turn or a cancelled/truncated response while playing flushes playback.

(() => {
  'use strict';

  // ---- DOM ----
  const callBtn = document.getElementById('callBtn');
  const pttBtn = document.getElementById('pttBtn');
  const statusEl = document.getElementById('status');
  const stateEl = document.getElementById('modelState');
  const timerEl = document.getElementById('timer');
  const convEl = document.getElementById('conversation');
  const logEl = document.getElementById('log');
  const vuFill = document.getElementById('vuFill');
  const micDot = document.getElementById('micDot');
  const voiceSel = document.getElementById('voiceSel');
  const refField = document.getElementById('refField');
  const refFile = document.getElementById('refFile');
  const modeSel = document.getElementById('modeSel');
  const turnDetSel = document.getElementById('turnDetSel');
  const lockNote = document.getElementById('lockNote');

  // ---- Config ----
  const WS_BASE =
    (window.DUPLEX_WS_BASE && window.DUPLEX_WS_BASE.trim()) ||
    (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.hostname + ':8099';
  const MODEL = 'openbmb/MiniCPM-o-4_5';
  const WS_URL = `${WS_BASE}/v1/realtime?duplex=1&model=${encodeURIComponent(MODEL)}`;

  const TARGET_SR = 16000;     // mic upload rate
  const PLAYBACK_SR = 24000;   // model audio rate
  const SEND_INTERVAL_MS = 200;
  const MAX_RECONNECTS = 4;

  // ---- State ----
  let ws = null;
  let micStream = null;
  let micCtx = null;
  let micNode = null;
  let playCtx = null;
  let ttsNode = null;
  let running = false;          // call is up (WS + audio graph)
  let isPlaying = false;
  let pendingPCM = [];
  let sendTimer = null;
  let timerTimer = null;
  let captureSR = TARGET_SR;
  let callStart = 0;
  let mode = 'full';            // full | half | turn
  let micGateOpen = true;       // in half/turn, only true while button held
  let recording = false;        // turn mode: currently capturing an utterance
  let reconnects = 0;
  let manualStop = false;
  let sessionConfig = null;     // cached so reconnect re-applies voice/mode

  // transcript turn tracking
  let liveAsst = null;          // {el, text} for the assistant turn in progress
  let liveUser = null;

  function log(msg, cls) {
    const t = new Date().toLocaleTimeString();
    const line = `[${t}] ${msg}`;
    const span = document.createElement('div');
    if (cls) span.className = cls;
    span.textContent = line;
    logEl.insertBefore(span, logEl.firstChild);
  }
  function setStatus(s) { statusEl.textContent = s; }
  function setModelState(s) {
    stateEl.textContent = s;
    stateEl.className = 'badge ' + (s === 'speaking' ? 'speaking' : s === 'listening' ? 'listening' : 'idle');
  }
  function setMicLive(on) {
    micDot.classList.toggle('live', !!on);
    if (!on) vuFill.style.width = '0%';
  }

  // ---- transcript turns ----
  function clearConvPlaceholder() {
    const e = convEl.querySelector('.empty');
    if (e) e.remove();
  }
  function appendTurn(who) {
    clearConvPlaceholder();
    const turn = document.createElement('div');
    turn.className = 'turn live ' + who;
    const w = document.createElement('div'); w.className = 'who'; w.textContent = who;
    const txt = document.createElement('div'); txt.className = 'text';
    turn.appendChild(w); turn.appendChild(txt);
    convEl.appendChild(turn);
    convEl.scrollTop = convEl.scrollHeight;
    return { el: turn, txt, text: '' };
  }
  function finalizeTurn(t) {
    if (t && t.el) t.el.classList.remove('live');
  }
  function asstDelta(d) {
    if (!liveAsst) liveAsst = appendTurn('assistant');
    liveAsst.text += d; liveAsst.txt.textContent = liveAsst.text;
    convEl.scrollTop = convEl.scrollHeight;
  }
  function asstDone() { finalizeTurn(liveAsst); liveAsst = null; }
  function userDelta(d) {
    if (!liveUser) liveUser = appendTurn('user');
    liveUser.text += d; liveUser.txt.textContent = liveUser.text;
    convEl.scrollTop = convEl.scrollHeight;
  }
  function userDone(full) {
    if (full && !liveUser) liveUser = appendTurn('user');
    if (full && liveUser) { liveUser.text = full; liveUser.txt.textContent = full; }
    finalizeTurn(liveUser); liveUser = null;
  }

  // ---- base64 helpers ----
  function int16ToBase64(int16) {
    const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
    let bin = '';
    const CH = 0x8000;
    for (let i = 0; i < bytes.length; i += CH) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
    return btoa(bin);
  }
  function base64ToInt16(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    return new Int16Array(bytes.buffer, 0, len >> 1);
  }
  function fileToDataURI(file) {
    return new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve(r.result);
      r.onerror = () => reject(new Error('could not read reference audio'));
      r.readAsDataURL(file);
    });
  }

  // ---- resampler (captureSR -> 16k) ----
  function resampleInt16(int16In, srIn, srOut) {
    if (srIn === srOut) return int16In;
    if (srIn > srOut) {
      // anti-alias downsample: box-average over each input window
      const step = srIn / srOut;
      const outLen = Math.floor(int16In.length / step);
      const out = new Int16Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const start = Math.floor(i * step);
        const end = Math.min(Math.floor((i + 1) * step), int16In.length);
        let sum = 0, n = 0;
        for (let j = start; j < end; j++) { sum += int16In[j]; n++; }
        out[i] = n ? (sum / n) | 0 : 0;
      }
      return out;
    }
    const ratio = srOut / srIn;
    const outLen = Math.floor(int16In.length * ratio);
    const out = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i / ratio;
      const i0 = Math.floor(pos);
      const i1 = Math.min(i0 + 1, int16In.length - 1);
      const frac = pos - i0;
      out[i] = (int16In[i0] * (1 - frac) + int16In[i1] * frac) | 0;
    }
    return out;
  }

  // ---- VU meter from captured Int16 ----
  function updateVU(int16) {
    let peak = 0;
    for (let i = 0; i < int16.length; i += 8) { const a = Math.abs(int16[i]); if (a > peak) peak = a; }
    const pct = Math.min(100, (peak / 32768) * 140);
    vuFill.style.width = pct.toFixed(0) + '%';
  }

  function flushMic() {
    if (!running || !ws || ws.readyState !== WebSocket.OPEN || pendingPCM.length === 0) return;
    let total = 0;
    for (const c of pendingPCM) total += c.length;
    const cat = new Int16Array(total);
    let off = 0;
    for (const c of pendingPCM) { cat.set(c, off); off += c.length; }
    pendingPCM = [];
    // Mic gate: in half/turn mode only stream while the button is held.
    if (!micGateOpen) return;
    const res = resampleInt16(cat, captureSR, TARGET_SR);
    ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: int16ToBase64(res) }));
  }

  // ---- playback / barge-in ----
  function feedPlayback(int16) { if (ttsNode) ttsNode.port.postMessage(int16); }
  function flushPlayback(reason) {
    if (ttsNode) ttsNode.port.postMessage({ type: 'clear' });
    if (isPlaying) log('barge-in: flush playback (' + reason + ')');
    isPlaying = false;
  }

  // ---- WS event handling ----
  function handleEvent(e) {
    switch (e.type) {
      case 'session.created': log('session.created'); reconnects = 0; break;
      case 'session.updated': break;
      case 'response.listen':
        setModelState('listening');
        if (mode === 'full') flushPlayback('listen');
        // Turn-based: a listen decision is a complete (empty) reply. Without
        // this the status would hang on "waiting for reply" since the model
        // emits no audio and may not send a prompt response.done.
        else if (mode === 'turn') setStatus('model listened — no reply (hold to record; try a fuller sentence)');
        break;
      case 'response.speak':
      case 'response.created':
        setModelState('speaking');
        break;
      case 'response.audio.delta': {
        const d = e.delta || (e.response && e.response.audio);
        if (d) { isPlaying = true; setModelState('speaking'); feedPlayback(base64ToInt16(d)); }
        break;
      }
      case 'response.audio_transcript.delta':
        if (e.delta) asstDelta(e.delta);
        break;
      case 'response.audio_transcript.done':
        asstDone();
        break;
      case 'conversation.item.input_audio_transcription.delta':
        if (e.delta) userDelta(e.delta);
        break;
      case 'conversation.item.input_audio_transcription.completed':
        userDone(e.transcript || null);
        break;
      case 'conversation.item.truncated':
        flushPlayback('truncated'); asstDone();
        break;
      case 'response.done': {
        const st = (e.response && e.response.status) || e.status;
        if (st === 'cancelled') flushPlayback('cancelled');
        asstDone();
        setModelState('listening');
        if (mode === 'turn') setStatus('idle (hold to record)');
        break;
      }
      case 'error':
        log('server error: ' + JSON.stringify(e.error || e.code || e), 'err');
        break;
      default: break;
    }
  }

  // ---- build the initial session.update payload (voice + turn detection) ----
  async function buildSessionConfig() {
    const session = { modalities: ['audio', 'text'] };
    if (turnDetSel.value === 'server_vad') session.turn_detection = { type: 'server_vad' };

    if (voiceSel.value === '__ref__') {
      const f = refFile.files && refFile.files[0];
      if (!f) throw new Error('select a reference audio file (or pick Default voice)');
      const dataURI = await fileToDataURI(f);     // data:audio/...;base64,...
      session.extra_body = { ref_audio: dataURI }; // adapter resolves before runtime open
      log('voice: cloning from reference audio "' + f.name + '"');
    } else {
      session.voice = voiceSel.value;              // named voice, e.g. "default"
      log('voice: ' + voiceSel.value);
    }
    // Full-duplex: ask the server to run continuous per-chunk generation (model
    // decides speak/listen) instead of waiting for an explicit response.create.
    if (mode === 'full') session.extra_body = Object.assign({}, session.extra_body, { auto_response: true });
    return session;
  }

  // ---- push-to-talk wiring ----
  function pttDown() {
    if (!running || mode === 'full') return;
    micGateOpen = true;
    recording = true;
    pttBtn.classList.add('held');
    setMicLive(true);
    setStatus(mode === 'turn' ? 'recording…' : 'talking…');
    // clear any input buffered from before this press
    try { ws.send(JSON.stringify({ type: 'input_audio_buffer.clear' })); } catch (_) {}
  }
  function pttUp() {
    if (!running || mode === 'full' || !recording) return;
    recording = false;
    pttBtn.classList.remove('held');
    micGateOpen = false;
    flushMic(); // send whatever is still pending while gate was open
    setMicLive(false);
    if (mode === 'turn') {
      // turn-based: finalize the utterance and ask for a reply
      try {
        ws.send(JSON.stringify({ type: 'input_audio_buffer.commit', final: true }));
        ws.send(JSON.stringify({ type: 'response.create' }));
        setStatus('sent — waiting for reply');
        log('turn committed -> response.create');
      } catch (_) {}
    } else {
      // half-duplex: stop streaming; model decides when to reply
      setStatus('idle (hold to talk)');
    }
  }
  pttBtn.addEventListener('mousedown', pttDown);
  pttBtn.addEventListener('mouseup', pttUp);
  pttBtn.addEventListener('mouseleave', () => { if (recording) pttUp(); });
  pttBtn.addEventListener('touchstart', (ev) => { ev.preventDefault(); pttDown(); }, { passive: false });
  pttBtn.addEventListener('touchend', (ev) => { ev.preventDefault(); pttUp(); }, { passive: false });

  // ---- timer ----
  function fmt(s) { const m = (s / 60) | 0, ss = s % 60; return String(m).padStart(2, '0') + ':' + String(ss).padStart(2, '0'); }
  function startTimer() { callStart = Date.now(); timerEl.textContent = '00:00';
    timerTimer = setInterval(() => { timerEl.textContent = fmt(((Date.now() - callStart) / 1000) | 0); }, 1000); }
  function stopTimer() { if (timerTimer) { clearInterval(timerTimer); timerTimer = null; } }

  // ---- config lock during call ----
  function lockConfig(lock) {
    [voiceSel, refFile, modeSel, turnDetSel].forEach((el) => { if (el) el.disabled = lock; });
    lockNote.style.display = lock ? 'block' : 'none';
  }

  // ---- start / stop ----
  async function startCall() {
    if (running) return;
    callBtn.disabled = true;
    manualStop = false;
    mode = modeSel.value;
    setStatus('connecting…');
    try {
      sessionConfig = await buildSessionConfig();

      playCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: PLAYBACK_SR });
      await playCtx.audioWorklet.addModule('static/ttsPlaybackProcessor.js');
      ttsNode = new AudioWorkletNode(playCtx, 'tts-playback-processor');
      ttsNode.port.onmessage = (ev) => { if (ev.data && ev.data.type === 'ttsPlaybackStopped') isPlaying = false; };
      ttsNode.connect(playCtx.destination);
      await playCtx.resume();

      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true,
                 autoGainControl: true, sampleRate: { ideal: TARGET_SR } },
      });
      // Capture at 16k directly so the browser does anti-aliased resampling
      // (naive JS downsample 48k->16k aliases -> garbled audio the model mishears).
      try { micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR }); }
      catch (_) { micCtx = new (window.AudioContext || window.webkitAudioContext)(); }
      captureSR = micCtx.sampleRate;
      await micCtx.audioWorklet.addModule('static/pcmWorkletProcessor.js');
      const src = micCtx.createMediaStreamSource(micStream);
      micNode = new AudioWorkletNode(micCtx, 'pcm-worklet-processor');
      micNode.port.onmessage = (ev) => {
        if (!running) return;
        const i16 = new Int16Array(ev.data);
        if (micGateOpen) updateVU(i16);
        pendingPCM.push(i16);
      };
      src.connect(micNode);
      const sink = micCtx.createGain(); sink.gain.value = 0;
      micNode.connect(sink).connect(micCtx.destination);

      await openWS();

      running = true;
      // mode-specific gating: full streams continuously; half/turn gate on the PTT button.
      micGateOpen = (mode === 'full');
      pttBtn.style.display = (mode === 'full') ? 'none' : 'inline-block';
      pttBtn.textContent = (mode === 'turn') ? 'Hold to record' : 'Hold to talk';
      setMicLive(mode === 'full');

      sendTimer = setInterval(flushMic, SEND_INTERVAL_MS);
      callBtn.textContent = 'Hang up';
      callBtn.classList.add('active');
      callBtn.disabled = false;
      lockConfig(true);
      startTimer();
      setStatus('in call (' + mode + ', captureSR=' + captureSR + '->16k)');
      setModelState('listening');
      log('call started — mode=' + mode);
    } catch (err) {
      const m = err && err.message ? err.message : err;
      log('start failed: ' + m, 'err');
      setStatus('error: ' + m);
      await stopCall();
      callBtn.disabled = false;
    }
  }

  function openWS() {
    return new Promise((resolve, reject) => {
      ws = new WebSocket(WS_URL);
      ws.binaryType = 'arraybuffer';
      let opened = false;
      ws.onopen = () => {
        opened = true;
        ws.send(JSON.stringify({ type: 'session.update', session: sessionConfig }));
        log('ws open -> session.update (turn_detection=' +
            (turnDetSel.value === 'server_vad' ? 'server_vad' : 'model-driven') + ')');
        resolve();
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') return;
        let e; try { e = JSON.parse(ev.data); } catch (_) { return; }
        handleEvent(e);
      };
      ws.onerror = () => { if (!opened) reject(new Error('ws connect failed: ' + WS_URL)); };
      ws.onclose = () => {
        if (!running || manualStop) { log('ws closed'); return; }
        log('ws closed unexpectedly', 'err');
        tryReconnect();
      };
    });
  }

  function tryReconnect() {
    if (manualStop || !running) return;
    if (reconnects >= MAX_RECONNECTS) { log('giving up after ' + reconnects + ' reconnects', 'err'); stopCall(); return; }
    reconnects += 1;
    const delay = Math.min(4000, 500 * Math.pow(2, reconnects - 1));
    setStatus('reconnecting (' + reconnects + ')…');
    flushPlayback('reconnect');
    setTimeout(() => {
      if (manualStop || !running) return;
      openWS().then(() => {
        log('reconnected');
        setStatus('in call (' + mode + ')');
      }).catch((e) => { log('reconnect failed: ' + e.message, 'err'); tryReconnect(); });
    }, delay);
  }

  async function stopCall() {
    manualStop = true;
    running = false;
    recording = false;
    pttBtn.classList.remove('held');
    pttBtn.style.display = 'none';
    callBtn.classList.remove('active');
    callBtn.textContent = 'Start call';
    if (sendTimer) { clearInterval(sendTimer); sendTimer = null; }
    stopTimer();
    pendingPCM = [];
    flushPlayback('hangup');
    asstDone(); userDone(null);
    try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch (_) {}
    ws = null;
    try { if (micNode) micNode.disconnect(); } catch (_) {}
    try { if (micStream) micStream.getTracks().forEach((tr) => tr.stop()); } catch (_) {}
    try { if (micCtx) await micCtx.close(); } catch (_) {}
    try { if (playCtx) await playCtx.close(); } catch (_) {}
    micNode = null; micStream = null; micCtx = null; ttsNode = null; playCtx = null;
    setMicLive(false);
    setStatus('idle');
    setModelState('idle');
    lockConfig(false);
    log('call ended');
  }

  // ---- UI wiring ----
  voiceSel.addEventListener('change', () => {
    refField.style.display = (voiceSel.value === '__ref__') ? 'flex' : 'none';
  });
  callBtn.addEventListener('click', () => { if (running) stopCall(); else startCall(); });

  setModelState('idle');
  setStatus('idle — WS ' + WS_URL);
  log('ready. WS target: ' + WS_URL);
})();
