
"""

Author: Justin E.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

C = 3.0e8  # speed of light (m/s)


def ask_float(prompt: str, default: float) -> float:
    """Prompts user and converts input safely to float."""
    raw = input(f"{prompt} [default={default}]: ").strip().replace(",", "")
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print("  Invalid input → using default.")
        return default

def safe_freq(name: str, f: float, fallback: float) -> float:
    """Ensure frequency is finite and > 0."""
    if f is None or not np.isfinite(f) or f <= 0.0:
        print(f"  {name} must be > 0; using {fallback}.")
        return fallback
    return f

def omega(f):return 2 * math.pi * f    # ω = 2πf
def k(f):return 2 * math.pi * f / C              # k = 2π/λ = 2πf/c
def to_rad(deg): return deg * math.pi / 180.0
def wave(A, kval, wval, x, t, phi): return A * np.sin(kval * x - wval * t + phi)  # D = A sin(kx − ωt + φ)


print("\n=== Signal Interference (Justin E.) === (it is advised use default options)\n")

# Signal 1 input
A1 = ask_float("Amplitude A1 (V)", 2.0)
f1 = safe_freq("f1", ask_float("Frequency f1 (Hz)", 1.0e6), 1.0e6)
phi1_deg = ask_float("Phase angle φ1 (degrees)", 0.0)


A2 = ask_float("Amplitude A2 (V)", 1.5)
f2 = safe_freq("f2", ask_float("Frequency f2 (Hz)", 1.2e6), 1.2e6)
phi2_deg = ask_float("Phase angle φ2 (degrees)", 45.0)


x1 = ask_float("Fixed distance x1 (m) for time-domain plot", 0.0)
t1 = ask_float("Fixed time t1 (s) for distance-domain plot", 2e-6)


phi1 = to_rad(phi1_deg)
phi2 = to_rad(phi2_deg)
w1, w2 = omega(f1), omega(f2)
k1, k2 = k(f1), k(f2)
lam1, lam2 = C / f1, C / f2


print(f"\nParameters:")
print(f"  ω1={w1:.3e} rad/s,  k1={k1:.3e} rad/m,  λ1={lam1:.0f} m")
print(f"  ω2={w2:.3e} rad/s,  k2={k2:.3e} rad/m,  λ2={lam2:.0f} m")

# --- Distance-domain Plot (at fixed t1) ---
x = np.linspace(-500, 500, 1500)  # fixed range (matches your preferred style)
t_fixed = np.full_like(x, t1)
D1 = wave(A1, k1, w1, x, t_fixed, phi1)
D2 = wave(A2, k2, w2, x, t_fixed, phi2)
Dsum = D1 + D2

plt.figure(figsize=(8, 4))
plt.plot(x, D1,   label=f"D1(x) λ1={int(lam1)} m", color='tab:green')
plt.plot(x, D2,   label=f"D2(x) λ2={int(lam2)} m", color='tab:red')
plt.plot(x, Dsum, label="Dsum(x) = D1 + D2",       color='tab:blue')
plt.title(f"The Distance domain for a fixed t value = {t1:g} s")
plt.xlabel("DISTANCE x (m)")
plt.ylabel("VOLTAGE (V)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("distance_domain.png", dpi=160)

# --- Time-domain Plot (at fixed x1) ---
# Use ~3 periods of the higher frequency so oscillations are visible
fmax = max(f1, f2)
Tmax = 1.0 / fmax
t = np.linspace(0.0, 3.0 * Tmax, 1500)
x_fixed = np.full_like(t, x1)
D1t = wave(A1, k1, w1, x_fixed, t, phi1)
D2t = wave(A2, k2, w2, x_fixed, t, phi2)
Dsumt = D1t + D2t

plt.figure(figsize=(8, 4))
plt.plot(t, D1t,   label="D1(t)",               color='tab:green')
plt.plot(t, D2t,   label="D2(t)",               color='tab:red')
plt.plot(t, Dsumt, label="Dsum(t) = D1 + D2",   color='tab:blue')
plt.title(f"The Time domain for a fixed x value = {x1:g} m")
plt.xlabel("TIME t (s)")
plt.ylabel("VOLTAGE (V)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("time_domain.png", dpi=160)

print("\nSaved: distance_domain.png, time_domain.png")
plt.show()

