"""Physical constants, noise laws and the guardrails no design may cross.

Rules enforced here (each raises PhysicsViolation):
  - light out of a passive optical stage never exceeds light in
  - an analog stage cannot be set faster than its parts allow (TIA bandwidth law)
  - shot noise is always on: it is a property of light, not a setting
"""
import math

Q = 1.602176634e-19      # electron charge, C
KB = 1.380649e-23        # Boltzmann, J/K
H = 6.62607015e-34       # Planck, J s
C = 299_792_458.0        # speed of light, m/s
T_ROOM = 300.0           # K


class PhysicsViolation(Exception):
    pass


def require(cond: bool, msg: str):
    if not cond:
        raise PhysicsViolation(msg)


def shot_noise_var(current_a, bandwidth_hz):
    """Variance (A^2) of shot noise: 2 q I B. Applies to photocurrent and dark current alike."""
    return 2.0 * Q * current_a * bandwidth_hz


def johnson_noise_var(resistance_ohm, bandwidth_hz, temp_k=T_ROOM):
    """Current-noise variance (A^2) of a resistor: 4 k T B / R."""
    return 4.0 * KB * temp_k * bandwidth_hz / resistance_ohm


def opamp_output_noise_var(en_v_rthz, c_in_f, c_f_f, gbw_hz):
    """Output voltage-noise variance from op-amp voltage noise in a transimpedance stage.
    Above the feedback pole the noise gain is 1 + Cin/Cf, and it stays there until the
    op-amp runs out of gain at GBW / (1 + Cin/Cf); the feedback capacitor does not filter it.
    With no filter before the ADC, all of it is sampled: en^2 * NG * GBW * pi/2."""
    ng = 1 + c_in_f / c_f_f
    return en_v_rthz ** 2 * ng * gbw_hz * math.pi / 2


def tia_max_bandwidth(gbw_hz, rf_ohm, cin_f):
    """Fastest stable bandwidth of a transimpedance amplifier: sqrt(GBW / (2 pi Rf Cin))."""
    return math.sqrt(gbw_hz / (2 * math.pi * rf_ohm * cin_f))


def noise_bandwidth(f3db_hz):
    """Equivalent noise bandwidth of a single-pole filter."""
    return f3db_hz * math.pi / 2


def lambert_share(det_area, h, dist2):
    """Share of a small glowing (Lambertian) patch's power landing on a small detector
    facing it at height h, distance sqrt(dist2): A_det * h^2 / (pi r^4)."""
    return det_area * h ** 2 / (math.pi * dist2 ** 2)


def photon_energy(wavelength_m):
    return H * C / wavelength_m


def abbe_limit(wavelength_m, na):
    """Smallest feature a lens of numerical aperture `na` can resolve."""
    require(0 < na <= 1.0, f"numerical aperture {na} impossible in air")
    return wavelength_m / (2 * na)


def check_passive(p_in, p_out, stage: str):
    require(p_out <= p_in * (1 + 1e-9), f"{stage}: {p_out:.3e} W out > {p_in:.3e} W in (energy from nowhere)")


def check_bandwidth(requested_hz, max_hz, stage: str):
    require(requested_hz <= max_hz * (1 + 1e-9),
            f"{stage}: asks for {requested_hz:.3g} Hz, parts allow at most {max_hz:.3g} Hz")


def shot_limited_bits(signal_electrons):
    """Bits of precision allowed by shot noise alone for a full-scale signal of N electrons.
    SNR = sqrt(N); bits = log2(SNR). No design can beat this for the given light."""
    return 0.5 * math.log2(max(signal_electrons, 1.0))
