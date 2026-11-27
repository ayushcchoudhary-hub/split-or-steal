// ============================================================================
// Split or Steal — 3D Apartment Replay Theater Audio Engine
// 1. Subtle Web Audio apartment room tone + noir thought dim + reveal cues
// 2. Character-distinct Web Speech API preview + ready `audio_url` playback hook
// ============================================================================

const VOICE_PROFILES = {
  oscar_wilde: { pitch: 1.08, rate: 0.98, preferredLang: 'en-GB' },
  warren_buffett: { pitch: 0.86, rate: 0.92, preferredLang: 'en-US' },
  richard_nixon: { pitch: 0.76, rate: 0.90, preferredLang: 'en-US' },
  machiavelli: { pitch: 0.94, rate: 1.02, preferredLang: 'en-GB' },
  gandhi: { pitch: 1.04, rate: 0.88, preferredLang: 'en-IN' },
  van_gogh: { pitch: 1.12, rate: 0.96, preferredLang: 'en-GB' },
};

const AMBIENT_MOODS = {
  intro:   { f1: 73.42, f2: 110.00, cut: 195, air: 0.055 }, // Warm apartment D2 + A2
  speech:  { f1: 82.41, f2: 123.47, cut: 220, air: 0.065 }, // Conversational E2 + B2
  thought: { f1: 58.27, f2: 87.31,  cut: 130, air: 0.032 }, // Hushed noir Bb1 + F2
  reveal:  { f1: 65.41, f2: 98.00,  cut: 175, air: 0.050 }, // Dramatic C2 + G2
};

export class TheaterAudio {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.osc1 = null;
    this.osc2 = null;
    this.filter = null;
    this.airGain = null;

    this.ambientEnabled = false;
    this.voiceEnabled = false;
    this.playbackRate = 1.0;
    this.currentMood = 'intro';

    this.activeAudioEl = null;
    this.speechToken = 0;
    this.cachedVoices = [];

    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      const loadVoices = () => {
        try {
          this.cachedVoices = window.speechSynthesis.getVoices() || [];
        } catch (_) {
          this.cachedVoices = [];
        }
      };
      loadVoices();
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
  }

  _initContext() {
    if (this.ctx) return true;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return false;

    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.0001;
    this.master.connect(this.ctx.destination);

    // Warm apartment low-passed harmonic bed
    this.filter = this.ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.frequency.value = AMBIENT_MOODS.intro.cut;

    const droneGain = this.ctx.createGain();
    droneGain.gain.value = 0.042;

    this.osc1 = this.ctx.createOscillator();
    this.osc1.type = 'triangle';
    this.osc1.frequency.value = AMBIENT_MOODS.intro.f1;

    this.osc2 = this.ctx.createOscillator();
    this.osc2.type = 'sine';
    this.osc2.frequency.value = AMBIENT_MOODS.intro.f2;

    this.osc1.connect(this.filter);
    this.osc2.connect(this.filter);
    this.filter.connect(droneGain);
    droneGain.connect(this.master);

    this.osc1.start();
    this.osc2.start();

    // Soft warm room air / subtle analog floor texture
    const len = this.ctx.sampleRate * 3;
    const buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) {
      const dust = Math.random() > 0.9992 ? (Math.random() * 2 - 1) * 0.18 : 0;
      data[i] = (Math.random() * 2 - 1) * 0.02 + dust;
    }

    const noise = this.ctx.createBufferSource();
    noise.buffer = buf;
    noise.loop = true;

    const band = this.ctx.createBiquadFilter();
    band.type = 'bandpass';
    band.frequency.value = 920;
    band.Q.value = 0.7;

    this.airGain = this.ctx.createGain();
    this.airGain.gain.value = AMBIENT_MOODS.intro.air;

    noise.connect(band);
    band.connect(this.airGain);
    this.airGain.connect(this.master);
    noise.start();

    return true;
  }

  toggleAmbient() {
    if (!this.ambientEnabled) {
      if (!this._initContext()) return false;
      if (this.ctx.state === 'suspended') {
        this.ctx.resume().catch(() => {});
      }
      this.ambientEnabled = true;
      const now = this.ctx.currentTime;
      this.master.gain.cancelScheduledValues(now);
      this.master.gain.setTargetAtTime(0.6, now, 0.45);
      this.setMood(this.currentMood, true);
      return true;
    }

    this.ambientEnabled = false;
    if (this.ctx) {
      const now = this.ctx.currentTime;
      this.master.gain.cancelScheduledValues(now);
      this.master.gain.setTargetAtTime(0.0001, now, 0.3);
    }
    return false;
  }

  setMood(mood, force = false) {
    if (!force && mood === this.currentMood) return;
    this.currentMood = mood;
    if (!this.ctx || !this.ambientEnabled) return;

    const m = AMBIENT_MOODS[mood] || AMBIENT_MOODS.intro;
    const now = this.ctx.currentTime;
    this.osc1.frequency.setTargetAtTime(m.f1, now, 0.75);
    this.osc2.frequency.setTargetAtTime(m.f2, now, 0.75);
    this.filter.frequency.setTargetAtTime(m.cut, now, 0.65);
    this.airGain.gain.setTargetAtTime(m.air, now, 0.65);
  }

  playCardFlipCue(outcome = 'split') {
    if (!this.ctx || !this.ambientEnabled) return;
    const now = this.ctx.currentTime;

    // Subtle harmonic sting matching outcome: split (warm), steal (sharp), crash (low)
    const freqs =
      outcome === 'mutual_split'
        ? [261.63, 329.63, 392.0]
        : outcome === 'mutual_crash'
          ? [130.81, 138.59]
          : [196.0, 233.08, 293.66];

    freqs.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = outcome === 'mutual_split' ? 'triangle' : 'sawtooth';
      osc.frequency.setValueAtTime(freq, now + idx * 0.045);

      const filt = this.ctx.createBiquadFilter();
      filt.type = 'lowpass';
      filt.frequency.value = outcome === 'mutual_split' ? 900 : 520;

      gain.gain.setValueAtTime(0.0001, now + idx * 0.045);
      gain.gain.linearRampToValueAtTime(0.032, now + idx * 0.045 + 0.06);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + idx * 0.045 + 1.15);

      osc.connect(filt);
      filt.connect(gain);
      gain.connect(this.master);

      osc.start(now + idx * 0.045);
      osc.stop(now + idx * 0.045 + 1.2);
    });
  }

  setPlaybackRate(rate) {
    this.playbackRate = rate;
    if (this.activeAudioEl) {
      this.activeAudioEl.playbackRate = rate;
    }
  }

  toggleVoice() {
    this.voiceEnabled = !this.voiceEnabled;
    if (!this.voiceEnabled) {
      this.stopSpeech();
    }
    return this.voiceEnabled;
  }

  stopSpeech() {
    this.speechToken += 1;
    if (this.activeAudioEl) {
      try {
        this.activeAudioEl.pause();
        this.activeAudioEl.currentTime = 0;
      } catch (_) {}
      this.activeAudioEl = null;
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch (_) {}
    }
  }

  _pickVoice(profile) {
    if (!this.cachedVoices.length) {
      try {
        this.cachedVoices = window.speechSynthesis.getVoices() || [];
      } catch (_) {
        this.cachedVoices = [];
      }
    }
    if (!this.cachedVoices.length) return null;

    const exact = this.cachedVoices.find(
      (v) => v.lang && v.lang.toLowerCase().startsWith(profile.preferredLang.toLowerCase())
    );
    if (exact) return exact;

    const english = this.cachedVoices.find((v) => v.lang && v.lang.toLowerCase().startsWith('en'));
    return english || this.cachedVoices[0] || null;
  }

  /**
   * Speaks a dialogue or thought beat when Voice is enabled.
   * Prefers pre-recorded `audioUrl` if provided; falls back to Web Speech API.
   * Calls `onEnd()` when playback completes (only if still the active token).
   */
  speakBeat({ text, characterId, audioUrl = null, isThought = false, onEnd = null }) {
    this.stopSpeech();
    if (!this.voiceEnabled || !text) return false;

    const token = this.speechToken;

    // 1. Pre-rendered voice clip hook (`audio_url` from highlights.json)
    if (audioUrl) {
      try {
        const audio = new Audio(audioUrl);
        audio.playbackRate = this.playbackRate;
        audio.volume = isThought ? 0.72 : 1.0;
        this.activeAudioEl = audio;
        audio.onended = () => {
          if (this.speechToken === token) {
            this.activeAudioEl = null;
            if (onEnd) onEnd();
          }
        };
        audio.onerror = () => {
          if (this.speechToken === token) {
            this.activeAudioEl = null;
            this._speakSynth({ text, characterId, isThought, token, onEnd });
          }
        };
        audio.play().catch(() => {
          if (this.speechToken === token) {
            this.activeAudioEl = null;
            this._speakSynth({ text, characterId, isThought, token, onEnd });
          }
        });
        return true;
      } catch (_) {
        // Fall through to Web Speech API
      }
    }

    // 2. Web Speech API preview fallback
    return this._speakSynth({ text, characterId, isThought, token, onEnd });
  }

  _speakSynth({ text, characterId, isThought, token, onEnd }) {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      return false;
    }

    try {
      const utter = new SpeechSynthesisUtterance(text);
      const profile = VOICE_PROFILES[characterId] || {
        pitch: 1.0,
        rate: 0.96,
        preferredLang: 'en-US',
      };

      const voice = this._pickVoice(profile);
      if (voice) utter.voice = voice;

      utter.pitch = isThought ? Math.max(0.65, profile.pitch - 0.06) : profile.pitch;
      utter.rate = Math.min(1.8, profile.rate * this.playbackRate * (isThought ? 1.05 : 1.0));
      utter.volume = isThought ? 0.65 : 0.95;

      utter.onend = () => {
        if (this.speechToken === token && onEnd) onEnd();
      };
      utter.onerror = () => {
        if (this.speechToken === token && onEnd) onEnd();
      };

      window.speechSynthesis.speak(utter);
      return true;
    } catch (_) {
      return false;
    }
  }
}
