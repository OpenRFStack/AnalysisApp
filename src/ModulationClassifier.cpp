/*
========================================================================
Project: OpenRFStack
Author:  Brendan Michaud
Year:    2026
Part of OpenRFStack (https://github.com/OpenRFStack)

Licensed under the Personal Use License.
Do not use for commercial, organizational, or military purposes.
Contact author for permission: https://github.com/OpenRFStack
========================================================================
*/
#include "ModulationClassifier.hpp"
#include <au/units/hertz.hh>
#include <au/units/seconds.hh>
#include <cmath>
#include <spdlog/spdlog.h>

namespace analysis {

// ── Internal helpers ──────────────────────────────────────────────────────────

bool ModulationClassifier::isAnalog(const SignalFeatures& f) const
{
    // High envelope variance strongly suggests AM family
    if (f.envelope_variance_norm > 0.25 && f.fm_deviation_hz < f.bandwidth_hz / 4.0)
        return true;
    // AM with a carrier at any depth (m = 0.5 has var/mean² ≈ 0.125)
    if (f.carrier_power_fraction > 0.4 && f.am_index > 0.1)
        return true;
    // Wide FM deviation and low envelope variance suggests FM
    if (f.fm_deviation_hz > f.bandwidth_hz * 0.1 && f.envelope_variance_norm < 0.1)
        return true;
    // PM: moderate phase std, very constant envelope, low freq std
    if (f.inst_phase_std > 0.3
        && f.envelope_variance_norm < 0.15
        && f.inst_freq_std_hz < f.bandwidth_hz * 0.1)
        return true;
    return false;
}

bool ModulationClassifier::hasDigitalEvidence(const SignalFeatures& f) const
{
    // ANA-08: band-limited PSK has a large inst-frequency spread (phase steps)
    // and a modest envelope variance, so isAnalog() called QPSK/8-PSK "FM_WB".
    // A cyclostationary symbol line is the structural tell — analog voice has
    // none. The line must fit the occupied bandwidth (linear: OBW ≈ Rs(1+β);
    // FSK: a few Rs): a tone-modulated FM/AM test signal or CTCSS gives a line
    // far below the bandwidth. A strong carrier means AM with a tone.
    if (f.fsk_detected) return true;
    const double rs = f.symbol_rate_sps.in(au::hertz);
    const double bw = f.bandwidth_hz.in(au::hertz);
    if (rs <= 0.0 || bw <= 0.0) return false;
    if (f.carrier_power_fraction > 0.4) return false;
    // Tone-modulated FM: the "symbol" line is a Bessel line of one tone.
    if (f.inst_freq_line_fraction > 0.5) return false;
    return rs >= 0.2 * bw && rs <= 1.5 * bw;
}

bool ModulationClassifier::isConstantEnvelope(const SignalFeatures& f) const
{
    return f.envelope_variance_norm < 0.05;
}

// ── Analog classification ─────────────────────────────────────────────────────

std::string ModulationClassifier::classifyAnalog(const SignalFeatures& f,
                                                   double& index_out) const
{
    // FM family: constant envelope + large deviation
    if (f.fm_deviation_hz > f.bandwidth_hz * 0.1 && f.envelope_variance_norm < 0.1) {
        index_out = f.fm_deviation_hz.in(au::hertz);
        if (f.fm_deviation_hz > au::hertz(50000.0)) return "FM_WB";
        return "FM_NB";
    }

    // PM family
    if (f.inst_phase_std > 0.3
        && f.envelope_variance_norm < 0.15
        && f.inst_freq_std_hz < f.bandwidth_hz * 0.1) {
        index_out = f.inst_phase_std;
        return "PM";
    }

    // AM family
    index_out = f.am_index;

    // SSB: asymmetric spectrum
    if (f.am_index > 0.1 && f.spectral_symmetry < 0.5) {
        // Determine USB vs LSB from inst frequency centroid direction
        if (f.inst_freq_mean_hz.in(au::hertz) > 0) return "SSB_USB";
        return "SSB_LSB";
    }

    // DSB-SC: suppressed carrier, very small am_index
    if (f.am_index < 0.05 && f.envelope_variance_norm > 0.25)
        return "AM_DSB_SC";

    // CW / OOK: high flatness or very burst-like
    if (f.spectral_flatness > 0.8 || (f.is_burst && f.burst_duty_cycle < 0.6)) {
        index_out = 0;
        return "CW";
    }

    // Default: DSB with carrier
    return "AM_DSB_LC";
}

// ── Digital classification ────────────────────────────────────────────────────
//
// ANA-08: the old thresholds were wrong for the textbook values themselves
// (QPSK was matched at C40 ≈ −2 — that is BPSK; |C40| of QPSK is 1; 32-QAM at
// −0.5/−0.5) and they used Re{C40}, which depends on the carrier phase.
// Now: |C40| (rotation-invariant) and C42 of the symbol-spaced, noise-
// corrected samples (FeatureExtractor::computeDigitalFeatures), nearest
// theoretical point (Swami & Sadler 2000). FSK is decided first from the
// two-level instantaneous frequency.

std::string ModulationClassifier::classifyDigital(const SignalFeatures& f,
                                                    int& m_ary_out) const
{
    // 2-FSK / MSK / GMSK: constant envelope, two-level inst. frequency with
    // a transition-rate line (FeatureExtractor::computeDigitalFeatures).
    if (f.fsk_detected) {
        m_ary_out = 2;
        if (f.fsk_mod_index > 0.35 && f.fsk_mod_index < 0.65) return "GMSK";
        return "FSK";
    }

    if (!f.cumulants_symbol_spaced) {
        // Oversampled waveform: cumulants are scaled by the unknown pulse
        // shape — report the family only (rule confidence 0 → ONNX).
        m_ary_out = 0;
        return isConstantEnvelope(f) ? "PSK" : "UNKNOWN";
    }

    struct Ref { const char* name; int m; double c40, c42; };
    static constexpr Ref kRefs[] = {
        {"BPSK",  2,  2.000, -2.000},
        {"QPSK",  4,  1.000, -1.000},
        {"8PSK",  8,  0.000, -1.000},
        {"QAM16", 16, 0.680, -0.680},
        {"QAM32", 32, 0.190, -0.690},
        {"QAM64", 64, 0.619, -0.619},
    };
    const double c40 = std::hypot(f.c40_real, f.c40_imag);
    const Ref* best = nullptr;
    double best_d = 1e30;
    for (const auto& r : kRefs) {
        const double d = std::hypot(c40 - r.c40, f.c42 - r.c42);
        if (d < best_d) { best_d = d; best = &r; }
    }
    if (best_d > 0.5) {          // nothing close: not a known constellation
        m_ary_out = 0;
        return "UNKNOWN";
    }
    m_ary_out = best->m;
    return best->name;
}

// ── Main classify ─────────────────────────────────────────────────────────────

void ModulationClassifier::classify(const SignalFeatures& f,
                                     AnalysisResult& r) const
{
    // Copy spectral basics
    r.center_freq_hz  = f.center_freq_hz;
    r.bandwidth_hz    = f.bandwidth_hz;
    r.snr_db          = f.snr_db;
    r.symbol_rate_sps = f.symbol_rate_sps;

    // Layer 3: Channel structure (always available)
    r.is_burst           = f.is_burst;
    r.burst_duty_cycle   = f.burst_duty_cycle;
    r.is_fhss            = f.fhss_detected;
    r.is_dsss            = f.dsss_detected;
    r.is_tdma            = f.is_burst && (f.burst_period_ms.in(au::seconds) > 0);
    r.ofdm_subcarrier_spacing_hz = au::hertz(0.0);

    // Layer 4 defaults
    r.bit_rate_bps    = 0;
    r.line_code       = "";
    r.fec_detected    = false;
    r.has_sync_pattern= false;
    r.classified      = true;
    r.analog_index    = 0;
    r.m_ary           = 0;
    r.is_ofdm         = false;
    r.is_spread        = f.fhss_detected || f.dsss_detected;
    r.analog_modulation  = "";
    r.digital_modulation = "";

    // ── Step 1: SNR gate ──────────────────────────────────────────────────
    if (f.snr_db < 5.0) {
        r.classified    = false;
        r.reject_reason = "SNR too low";
        spdlog::debug("Classifier: SNR {:.1f} dB below threshold, unclassified", f.snr_db);
        return;
    }

    // ── Step 2: Chirp ─────────────────────────────────────────────────────
    if (f.chirp_detected) {
        r.digital_modulation = "CSS";
        r.m_ary = 2;
        spdlog::debug("Classifier: chirp detected (rate={:.0f} Hz/s) → CSS",
                      f.chirp_rate_hz_s);
        return;
    }

    // ── Step 3: OFDM ─────────────────────────────────────────────────────
    if (f.ofdm_detected) {
        r.digital_modulation = "OFDM";
        r.is_ofdm = true;
        if (f.ofdm_fft_size_est > 0)
            r.symbol_rate_sps = f.sample_rate_sps / f.ofdm_fft_size_est;
        r.ofdm_subcarrier_spacing_hz = (f.ofdm_fft_size_est > 0)
                                       ? f.sample_rate_sps / f.ofdm_fft_size_est
                                       : au::hertz(0.0);
        spdlog::debug("Classifier: OFDM Nfft={}", f.ofdm_fft_size_est);
        return;
    }

    // ── Step 4: FHSS ─────────────────────────────────────────────────────
    if (f.fhss_detected) {
        r.digital_modulation = "FHSS";
        r.is_spread = true;
        spdlog::debug("Classifier: FHSS detected hop_rate={:.0f} Hz",
                      f.fhss_hop_rate_hz);
        return;
    }

    // ── Step 5: DSSS ─────────────────────────────────────────────────────
    if (f.dsss_detected) {
        r.digital_modulation = "DSSS";
        r.is_spread = true;
        return;
    }

    // ── Step 6: Analog vs digital decision ───────────────────────────────
    if (!hasDigitalEvidence(f) && isAnalog(f)) {
        double idx = 0;
        r.analog_modulation = classifyAnalog(f, idx);
        r.analog_index      = idx;
        spdlog::debug("Classifier: analog → {}, index={:.3f}",
                      r.analog_modulation, idx);
        return;
    }

    // ── Step 7: Digital modulation ────────────────────────────────────────
    int m = 1;
    r.digital_modulation = classifyDigital(f, m);
    r.m_ary = m;

    // Line code heuristic for FSK
    if (r.digital_modulation == "FSK" || r.digital_modulation == "GMSK") {
        // Manchester encoding doubles the bandwidth relative to symbol rate
        const double bw_hz = f.bandwidth_hz.in(au::hertz);
        const double sr_hz = f.symbol_rate_sps.in(au::hertz);
        if (bw_hz > 0 && sr_hz > 0) {
            double ratio = bw_hz / sr_hz;
            if (ratio > 1.8 && ratio < 2.5)
                r.line_code = "Manchester";
        }
        if (r.line_code.empty()) r.line_code = "NRZ";
    }

    // ── Step 8/9: Bitstream traits ────────────────────────────────────────
    if (r.m_ary > 1 && r.symbol_rate_sps.in(au::hertz) > 0) {
        double bits_per_sym = std::log2(r.m_ary);
        r.bit_rate_bps = r.symbol_rate_sps.in(au::hertz) * bits_per_sym;
    }

    spdlog::debug("Classifier: digital → {} m_ary={} sr={:.0f} sps bit_rate={:.0f} bps",
                  r.digital_modulation, r.m_ary,
                  r.symbol_rate_sps.in(au::hertz), r.bit_rate_bps);
}

} // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
