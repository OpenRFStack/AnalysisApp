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
#pragma once
/**
 * @file FeatureExtractor.hpp
 * @brief Extracts a SignalFeatures vector from a block of IQ samples.
 *
 * Feature extraction covers:
 * - PSD computation (Welch), spectral flatness, symmetry
 * - Envelope statistics and instantaneous frequency/phase
 * - Higher-order cumulants (C40, C42, C41) for modulation discrimination
 * - Symbol rate estimation via cyclostationary analysis
 * - Structural detection: OFDM (CP autocorrelation), FHSS (spectrogram),
 *   DSSS (chip-rate peak), chirp (Wigner-Ville), burst (envelope gating)
 *
 * One FeatureExtractor per analysis pipeline (owns FFTW plans).
 * Thread-safe: extract() is const; FFTW plans are read-only after construction.
 */
#include "SignalFeatures.hpp"

#include <complex>
#include <vector>

#include <au/units/hertz.hh>

namespace analysis
{

/// @brief Extracts a SignalFeatures vector from raw IQ samples.
class FeatureExtractor
{
public:
    /**
     * @brief Construct the extractor and allocate FFTW plans.
     * @param fft_size FFT size for PSD and cyclostationary analysis.
     */
    explicit FeatureExtractor(int fft_size = 4096);
    ~FeatureExtractor();

    /**
     * @brief Extract all features from a block of IQ samples.
     * @param iq              Interleaved float32 samples (I,Q,I,Q,…).
     * @param sample_rate_sps Sample rate (samples/s).
     * @param center_freq_hz  Centre frequency (Hz); stored verbatim in the result.
     * @return Fully populated SignalFeatures.
     */
    SignalFeatures extract(const std::vector<float>& iq,
                           au::QuantityD<au::Hertz>  sample_rate_sps,
                           au::QuantityD<au::Hertz>  center_freq_hz) const;

private:
    int   fft_size_;
    void* fft_plan_  = nullptr;  ///< FFTW plan for PSD.
    void* fft_plan2_ = nullptr;  ///< FFTW plan for cyclostationary analysis.
    /// Hann window of fft_size_ (built once; was rebuilt per PSD / FHSS segment).
    std::vector<float> hann_;
    /// Plans for the symbol-rate FFT, one per power of two in
    /// [fft_size_, kMaxSymbolRateLen]. All planning happens in the constructor:
    /// the FFTW planner is not thread-safe and extract() runs on several
    /// AnalysisService workers at once (fftwf_execute_dft is thread-safe).
    std::vector<void*>              sym_plans_;
    std::vector<std::vector<float>> sym_windows_;  ///< Hann window per sym_plans_ entry.
    static constexpr int            kMaxSymbolRateLen = 65'536;
    int                             symPlanIndex(int n) const;

    void computePsd(const std::vector<std::complex<float>>& x, std::vector<float>& psd_out) const;

    /// Normalised cumulants of @p x. @p noise_power (per sample) is removed
    /// from the normaliser: Gaussian noise adds nothing to 4th-order
    /// cumulants, so C4x / (M21 − σ²)² is the signal's own value.
    void computeCumulants(const std::vector<std::complex<float>>& x,
                          double&                                 c40r,
                          double&                                 c40i,
                          double&                                 c41r,
                          double&                                 c42,
                          double                                  noise_power = 0.0) const;

    struct SymbolRateEst
    {
        double rate_hz       = 0.0;  ///< 0 = no line found.
        double carrier_hz    = 0.0;  ///< Carrier offset from the x⁴ line (valid if carrier_valid).
        bool   carrier_valid = false;
    };

    SymbolRateEst estimateSymbolRate(const std::vector<std::complex<float>>& x,
                                     au::QuantityD<au::Hertz>                sample_rate_sps) const;

    /// Carrier-corrected, symbol-spaced cumulants and the FSK discriminants.
    void computeDigitalFeatures(const std::vector<std::complex<float>>& x,
                                double                                  sample_rate,
                                const SymbolRateEst&                    sr_est,
                                double                                  noise_power,
                                double                                  band_lo_hz,
                                double                                  band_hi_hz,
                                SignalFeatures&                         f) const;

    bool detectOfdm(const std::vector<std::complex<float>>& x,
                    au::QuantityD<au::Hertz>                sample_rate_sps,
                    int&                                    fft_size_out,
                    double&                                 cp_ratio_out) const;

    bool detectFhss(const std::vector<std::complex<float>>& x,
                    au::QuantityD<au::Hertz>                sample_rate_sps,
                    au::QuantityD<au::Hertz>&               hop_rate_out) const;

    /// @param fi_rad instantaneous frequency (rad/sample), shared with the stats.
    bool detectChirp(const std::vector<float>& fi_rad,
                     au::QuantityD<au::Hertz>  sample_rate_sps,
                     double&                   chirp_rate_out) const;

    void computeInstFreqStats(const std::vector<float>& fi_rad,
                              au::QuantityD<au::Hertz>  sample_rate_sps,
                              au::QuantityD<au::Hertz>& mean,
                              au::QuantityD<au::Hertz>& std_dev) const;

    bool detectBurst(const std::vector<std::complex<float>>& x,
                     double&                                 duty_cycle,
                     double&                                 period_samples) const;
};

}  // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
