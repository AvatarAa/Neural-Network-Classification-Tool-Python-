import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid



# --- 1. Define Common Parameters ---

Ac = 1.0                # Carrier amplitude (V)

Fc = 100e3              # Carrier frequency (100 kHz)

Fm = 1e3                # Message frequency (1 kHz)

Am = 1.0                # Message amplitude (V)



Fs = 500e3              # Sampling frequency (500 kHz, must be > 2*Fc)

T_sim = 5 / Fm          # Total simulation time (5 cycles of Fm)

t = np.arange(0, T_sim, 1/Fs) # Time vector



# Message signal m(t)

m_t = Am * np.cos(2 * np.pi * Fm * t)



# --- 2. Phase Modulation (PM) Implementation ---

kp = np.pi / 2          # Phase sensitivity constant (rad/V)

phasedev = kp * Am      # Phase deviation (Delta_phi = 1.57 rad)



# Generate the PM signal

s_pm = Ac * np.cos(2 * np.pi * Fc * t + kp * m_t)

modulation_index_pm = phasedev # Beta_p



# --- PM PLOTS (Time and Frequency Domain) ---

plt.figure(figsize=(12, 8))



# Time Domain (Deliverable b)

plt.subplot(2, 2, 1)

plt.plot(t * 1e3, m_t)

plt.title('PM Modulating Signal, $m(t)$')

plt.xlabel('Time (ms)')

plt.ylabel('Amplitude (V)')

plt.grid(True)



plt.subplot(2, 2, 2)

plt.plot(t * 1e3, s_pm)

plt.title('PM Signal (Time Domain)')

plt.xlabel('Time (ms)')

plt.ylabel('Amplitude (V)')

plt.grid(True)



# Frequency Domain (Deliverable c)

N_pm = len(s_pm)

S_pm = np.fft.fft(s_pm)

frequencies_pm = np.fft.fftfreq(N_pm, 1/Fs)



# Plotting the single-sided spectrum for clarity

single_sided_mag_pm = 2 * np.abs(S_pm[:N_pm//2]) / N_pm

single_sided_freq_pm = frequencies_pm[:N_pm//2]



plt.subplot(2, 1, 2)

# Zooming in around the carrier (50 kHz to 150 kHz)

mask_pm = (single_sided_freq_pm >= 50e3) & (single_sided_freq_pm <= 150e3)

plt.plot(single_sided_freq_pm[mask_pm] / 1e3, single_sided_mag_pm[mask_pm])

plt.title('PM Signal (Frequency Domain - Zoomed FFT)')

plt.xlabel('Frequency (kHz)')

plt.ylabel('|S(f)| Magnitude')

plt.grid(True)

plt.tight_layout()

plt.show()



# --- 3. Frequency Modulation (FM) Implementation ---

kf = 5e3                # Frequency sensitivity constant (Hz/V)

freqdev = kf * Am       # Frequency deviation (Delta_f = 5 kHz)



# Numerical integration of the message signal: integral(m(t)) dt

integral_m_t = cumulative_trapezoid(m_t, t, initial=0)



# Generate the FM signal

# Phase term: phi(t) = 2*pi*kf * integral(m(t)) dt

phase_term_fm = 2 * np.pi * kf * integral_m_t

s_fm = Ac * np.cos(2 * np.pi * Fc * t + phase_term_fm)



# Modulation Index for FM

modulation_index_fm = freqdev / Fm # Beta_f = 5 kHz / 1 kHz = 5



# --- FM PLOTS (Time and Frequency Domain) ---

plt.figure(figsize=(12, 8))



# Time Domain (Deliverable b)

plt.subplot(2, 2, 1)

plt.plot(t * 1e3, m_t)

plt.title('FM Modulating Signal, $m(t)$')

plt.xlabel('Time (ms)')

plt.ylabel('Amplitude (V)')

plt.grid(True)



plt.subplot(2, 2, 2)

plt.plot(t * 1e3, s_fm)

plt.title('FM Signal (Time Domain)')

plt.xlabel('Time (ms)')

plt.ylabel('Amplitude (V)')

plt.grid(True)



# Frequency Domain (Deliverable c)

N_fm = len(s_fm)

S_fm = np.fft.fft(s_fm)

frequencies_fm = np.fft.fftfreq(N_fm, 1/Fs)



# Plotting the single-sided spectrum

single_sided_mag_fm = 2 * np.abs(S_fm[:N_fm//2]) / N_fm

single_sided_freq_fm = frequencies_fm[:N_fm//2]



plt.subplot(2, 1, 2)

# Zooming in around the carrier (50 kHz to 150 kHz)

mask_fm = (single_sided_freq_fm >= 50e3) & (single_sided_freq_fm <= 150e3)

plt.plot(single_sided_freq_fm[mask_fm] / 1e3, single_sided_mag_fm[mask_fm])

plt.title('FM Signal (Frequency Domain - Zoomed FFT)')

plt.xlabel('Frequency (kHz)')

plt.ylabel('|S(f)| Magnitude')

plt.grid(True)

plt.tight_layout()

plt.show()