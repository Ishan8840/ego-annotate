"""Original warm ambient score: slow chord swells, soft harmonics, diffuse reverb.

Requires numpy. No samples, bells, arpeggios, percussion, or external recordings.
"""
from pathlib import Path
import json
import wave
import numpy as np

SR, SECONDS = 48000, 30
N = SR * SECONDS
signal = np.zeros((N, 2), dtype=np.float64)
TAU = 2 * np.pi
# Dmaj9 -> Aadd9/C# -> Bm11 -> Gmaj9 -> Dmaj9: smooth shared tones.
chords = [[38, 50, 57, 61, 64, 66], [37, 49, 57, 59, 64, 69],
          [35, 47, 57, 61, 62, 66], [43, 50, 54, 57, 59, 66],
          [38, 50, 57, 61, 64, 66]]

def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)

for chord_index, chord in enumerate(chords):
    start = chord_index * 8 * SR
    if start >= N:
        break
    duration = min(12.0, SECONDS - start/SR)
    t = np.arange(round(duration*SR))/SR
    env = smooth(t/3.0) * smooth((duration-t)/4.5)
    for voice, midi in enumerate(chord):
        freq = 440 * 2**((midi-69)/12)
        pan = [.5, .38, .18, .78, .30, .67][voice]
        gain = [.065, .040, .029, .025, .021, .017][voice]
        tone = np.zeros(len(t))
        # Broad, gently moving oscillator ensemble; long attacks eliminate chimes.
        for layer, cents in enumerate([-4, 0, 4]):
            phase = TAU*freq*2**(cents/1200)*t + .6*voice + layer*1.9
            phase += .12*np.sin(TAU*(.11+.019*voice)*t + layer)
            tone += (np.sin(phase) + .20*np.sin(2*phase+.3) + .065*np.sin(3*phase))/3
        tone *= env*gain*(.98+.02*np.sin(TAU*.13*t+voice))
        signal[start:start+len(t),0] += tone*np.cos(pan*np.pi/2)
        signal[start:start+len(t),1] += tone*np.sin(pan*np.pi/2)

# Diffuse, deterministic stereo room response, with no audible repeated echoes.
rng = np.random.default_rng(701)
M = int(4.8*SR)
rt = np.arange(M)/SR
fft_size = 1 << (N+M-2).bit_length()
wet = np.zeros_like(signal)
for channel in range(2):
    impulse = rng.normal(0,1,M)*np.exp(-rt/0.95)*smooth(rt/.035)
    impulse = np.convolve(impulse, np.ones(23)/23, mode='same')
    impulse /= np.sqrt(np.sum(impulse**2))
    wet[:,channel] = np.fft.irfft(np.fft.rfft(signal[:,channel],fft_size)*
                                 np.fft.rfft(impulse,fft_size),fft_size)[:N]
signal = .83*signal + .17*wet
# Soft spectral roll-off and subsonic removal, preserving the low chord bed.
freqs = np.fft.rfftfreq(N,1/SR)
curve = (1-np.exp(-(freqs/35)**4)) / np.sqrt(1+(freqs/1800)**6)
for channel in range(2):
    signal[:,channel] = np.fft.irfft(np.fft.rfft(signal[:,channel])*curve,N)
time = np.arange(N)/SR
signal *= (smooth(time/2)*smooth((SECONDS-time)/3.8))[:,None]
signal -= signal.mean(axis=0)
peak = np.max(np.abs(signal))
signal *= 10**(-6/20)/peak
pcm = np.round(np.clip(signal,-1,1)*32767).astype('<i2')
out = Path(__file__).resolve().parents[1]/'artifacts/demo/ego-score.wav'
with wave.open(str(out),'wb') as f:
    f.setnchannels(2);f.setsampwidth(2);f.setframerate(SR);f.writeframes(pcm.tobytes())
# Record objective checks for clicks, clipping, and spectral balance.
mono = signal.mean(axis=1)
spectrum = np.abs(np.fft.rfft(mono))**2
stats = dict(style='Warm ambient chord bed', duration_s=SECONDS, sample_rate_hz=SR,
             peak_dbfs=float(20*np.log10(np.max(np.abs(signal)))),
             max_sample_step=float(np.max(np.abs(np.diff(signal,axis=0)))),
             energy_above_2khz_fraction=float(spectrum[freqs>2000].sum()/spectrum.sum()),
             fade_in_s=2, fade_out_s=3.8, chord_interval_s=8,
             percussion=False, external_recordings=False)
out.with_name('music-verification.json').write_text(json.dumps(stats,indent=2)+'\n')
assert np.isfinite(signal).all() and np.max(np.abs(signal)) < 1
assert stats['max_sample_step'] < .05
print(json.dumps(stats,indent=2))
print(out)
