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
// Audit regressions for FeatureExtractor / ModulationClassifier (ANA-03..05).
// Every test here was run against the unmodified code first.
#include "AnalysisResult.hpp"
#include "Config.hpp"
#include "FeatureExtractor.hpp"
#include "ModulationClassifier.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <au/units/hertz.hh>

namespace
{
constexpr double TWO_PI = 2.0 * std::numbers::pi_v<double>;
constexpr double SR     = 2'000'000.0;
constexpr int    NS     = 65'536;
using cd                = std::complex<double>;

std::vector<float> interleave(const std::vector<cd>& x)
{
    std::vector<float> iq;
    iq.reserve(x.size() * 2);
    for (auto& s : x)
    {
        iq.push_back((float)s.real());
        iq.push_back((float)s.imag());
    }
    return iq;
}

// OFDM with QPSK subcarriers and a cyclic prefix, starting `offset` samples
// into a symbol — a capture never starts on a symbol boundary.
std::vector<float> ofdm(int nfft, int ncp, int offset, uint32_t seed = 7)
{
    std::mt19937                       rng(seed);
    std::uniform_int_distribution<int> d(0, 3);
    const cd                           q[4] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
    std::vector<cd>                    out;
    while ((int)out.size() < NS + offset)
    {
        std::vector<cd> td(nfft);
        std::vector<cd> fd(nfft);
        for (int k = 1; k < nfft; ++k)
            fd[k] = q[d(rng)];
        for (int n = 0; n < nfft; ++n)
        {
            cd acc{0, 0};
            for (int k = 0; k < nfft; ++k)
                acc += fd[k] * std::polar(1.0, TWO_PI * k * n / nfft);
            td[n] = acc / (double)nfft;
        }
        for (int k = nfft - ncp; k < nfft; ++k)
            out.push_back(td[k]);
        for (int k = 0; k < nfft; ++k)
            out.push_back(td[k]);
    }
    return interleave({out.begin() + offset, out.begin() + offset + NS});
}

std::vector<float> tone(double f)
{
    std::vector<cd> x(NS);
    for (int i = 0; i < NS; ++i)
        x[i] = std::polar(1.0, TWO_PI * f * i / SR);
    return interleave(x);
}

std::vector<float> fm(double dev, double fmod)
{
    std::vector<cd> x(NS);
    double          ph = 0;
    for (int i = 0; i < NS; ++i)
    {
        ph += TWO_PI * dev / SR * std::cos(TWO_PI * fmod * i / SR);
        x[i] = std::polar(1.0, ph);
    }
    return interleave(x);
}

// Band-limited (raised-cosine, β = 0.35) PSK at symbol rate rs, carrier offset df.
std::vector<float> psk(int m, double rs, double df, uint32_t seed = 3)
{
    std::mt19937                       rng(seed);
    std::uniform_int_distribution<int> d(0, m - 1);
    const double                       sps  = SR / rs;
    const int                          nsym = (int)(NS / sps) + 40;
    std::vector<cd>                    sym(nsym);
    for (auto& s : sym)
        s = std::polar(1.0, TWO_PI * d(rng) / m + (m == 4 ? TWO_PI / 8 : 0.0));
    auto rc = [&](double t)
    {  // t in symbols
        const double b = 0.35;
        if (std::abs(t) < 1e-9)
            return 1.0;
        if (std::abs(std::abs(2 * b * t) - 1.0) < 1e-9)
            return std::numbers::pi / 4 * std::sin(std::numbers::pi * t) / (std::numbers::pi * t);
        return std::sin(std::numbers::pi * t) / (std::numbers::pi * t)
               * std::cos(std::numbers::pi * b * t) / (1 - 4 * b * b * t * t);
    };
    std::vector<cd> x(NS);
    for (int i = 0; i < NS; ++i)
    {
        const double t  = i / sps + 20;  // skip the filter warm-up
        const int    k0 = (int)t;
        cd           acc{0, 0};
        for (int k = k0 - 12; k <= k0 + 12; ++k)
            if (k >= 0 && k < nsym)
                acc += sym[k] * rc(t - k);
        x[i] = acc * std::polar(1.0, TWO_PI * df * i / SR);
    }
    return interleave(x);
}
}  // namespace

// ANA-03: the cyclic-prefix correlator summed windows at n = 0, L, 2L …, i.e.
// it assumed the capture starts exactly on an OFDM symbol boundary.
TEST(AuditFeatures, Ana03_OfdmDetectedAtAnyTimingOffset)
{
    analysis::FeatureExtractor fx(4096);
    for (int off : {0, 7, 29, 41, 63})
    {
        auto f = fx.extract(ofdm(64, 16, off), au::hertz(SR), au::hertz(0.0));
        EXPECT_TRUE(f.ofdm_detected) << "offset " << off;
        if (f.ofdm_detected)
        {
            EXPECT_EQ(f.ofdm_fft_size_est, 64) << "offset " << off;
            EXPECT_NEAR(f.ofdm_cp_ratio, 0.25, 1e-9) << "offset " << off;
        }
    }
}

// ANA-04: a narrowband signal is (nearly) periodic at every lag, so the CP
// score was ~1 for any Nfft — a CW carrier or an FM voice channel came out as
// "OFDM" (OFDM is decided before the analog/digital split).
TEST(AuditFeatures, Ana04_NarrowbandSignalsAreNotOfdm)
{
    analysis::FeatureExtractor     fx(4096);
    analysis::ModulationClassifier cls;

    struct Case
    {
        const char*        name;
        std::vector<float> iq;
    } cases[] = {
        {"CW tone", tone(100e3)},
        {"FM 5k/1k", fm(5e3, 1e3)},
        {"FM 75k/3k", fm(75e3, 3e3)},
    };

    for (auto& c : cases)
    {
        auto f = fx.extract(c.iq, au::hertz(SR), au::hertz(0.0));
        EXPECT_FALSE(f.ofdm_detected) << c.name;
        analysis::AnalysisResult r;
        cls.classify(f, r);
        EXPECT_NE(r.digital_modulation, "OFDM") << c.name;
    }
}

// ANA-05: the fourth-power branch divided the x⁴ spectral line by 4. For M-PSK
// x⁴ has a carrier line at 4·Δf and symbol-rate lines at 4·Δf ± Rs, so the
// estimate came out as Rs/4 (centred) or as the carrier offset (offset).
TEST(AuditFeatures, Ana05_SymbolRateOfBandLimitedPsk)
{
    analysis::FeatureExtractor fx(4096);

    struct Case
    {
        int    m;
        double rs, df;
    } cases[] = {
        {2, 100e3, 0.0},
        {4, 100e3, 0.0},
        {4, 100e3, 20e3},
        {2, 50e3, -15e3},
        {4, 250e3, 5e3},
    };

    for (auto& c : cases)
    {
        auto f = fx.extract(psk(c.m, c.rs, c.df), au::hertz(SR), au::hertz(0.0));
        EXPECT_NEAR(f.symbol_rate_sps.in(au::hertz), c.rs, c.rs * 0.05)
            << c.m << "-PSK Rs=" << c.rs << " df=" << c.df;
    }
}

TEST(AuditFeatures, Ana05_SymbolRateAt10dBSnr)
{
    analysis::FeatureExtractor fx(4096);
    auto                       iq = psk(4, 100e3, 12e3, 9);
    double                     p  = 0;
    for (float v : iq)
        p += (double)v * v;
    p /= (iq.size() / 2);
    std::mt19937                    rng(2);
    std::normal_distribution<float> g(0.f, (float)std::sqrt(p / 10.0 / 2.0));
    for (auto& v : iq)
        v += g(rng);
    auto f = fx.extract(iq, au::hertz(SR), au::hertz(0.0));
    EXPECT_NEAR(f.symbol_rate_sps.in(au::hertz), 100e3, 5e3);
}

// Guard: no line, no symbol rate (noise, CW, wideband FM).
TEST(AuditFeatures, Ana05_NoSymbolRateWithoutALine)
{
    analysis::FeatureExtractor      fx(4096);
    std::mt19937                    rng(1);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float>              noise(2 * NS);
    for (auto& v : noise)
        v = g(rng);
    EXPECT_EQ(fx.extract(noise, au::hertz(SR), au::hertz(0.0)).symbol_rate_sps.in(au::hertz), 0.0);
    EXPECT_EQ(fx.extract(tone(100e3), au::hertz(SR), au::hertz(0.0)).symbol_rate_sps.in(au::hertz),
              0.0);
}

// ── ANA-07 / 08 / 09 ─────────────────────────────────────────────────────────
namespace
{
struct Gen
{
    double sr;
    int    n = NS;

    // Raised-cosine (β 0.35) linear modulation, carrier offset df, phase 0.3.
    std::vector<cd> linear(const std::vector<cd>& cons, double rs, double df, uint32_t seed) const
    {
        std::mt19937                       rng(seed);
        std::uniform_int_distribution<int> d(0, (int)cons.size() - 1);
        const double                       sps  = sr / rs;
        const int                          nsym = (int)(n / sps) + 40;
        std::vector<cd>                    sym(nsym);
        for (auto& v : sym)
            v = cons[(size_t)d(rng)];
        auto rc = [](double t)
        {
            const double b = 0.35, pi = std::numbers::pi;
            if (std::abs(t) < 1e-9)
                return 1.0;
            if (std::abs(std::abs(2 * b * t) - 1.0) < 1e-9)
                return pi / 4 * std::sin(pi * t) / (pi * t);
            return std::sin(pi * t) / (pi * t) * std::cos(pi * b * t) / (1 - 4 * b * b * t * t);
        };
        std::vector<cd> x((size_t)n);
        for (int i = 0; i < n; ++i)
        {
            const double t  = i / sps + 20;
            const int    k0 = (int)t;
            cd           acc{0, 0};
            for (int k = k0 - 12; k <= k0 + 12; ++k)
                if (k >= 0 && k < nsym)
                    acc += sym[(size_t)k] * rc(t - k);
            x[(size_t)i] = acc * std::polar(1.0, TWO_PI * df * i / sr + 0.3);
        }
        return x;
    }

    // Continuous-phase 2-FSK, modulation index h.
    std::vector<cd> fsk(double rs, double h, double df, uint32_t seed) const
    {
        std::mt19937                       rng(seed);
        std::uniform_int_distribution<int> d(0, 1);
        const int                          sps = (int)(sr / rs);
        std::vector<cd>                    x((size_t)n);
        double                             ph = 0;
        int                                b  = 0;
        for (int i = 0; i < n; ++i)
        {
            if (i % sps == 0)
                b = d(rng);
            ph += TWO_PI * (df + (b ? 1 : -1) * h * rs / 2) / sr;
            x[(size_t)i] = std::polar(1.0, ph);
        }
        return x;
    }

    // FM, deviation dev, modulated by band-limited noise (voice-like).
    std::vector<cd> fmVoice(double dev, uint32_t seed) const
    {
        std::mt19937                     rng(seed);
        std::normal_distribution<double> g(0, 1);
        std::vector<cd>                  x((size_t)n);
        double                           ph = 0, m1 = 0, m2 = 0;
        const double a = std::exp(-TWO_PI * 1500.0 / sr);  // 2-pole ~1.5 kHz low-pass
        for (int i = 0; i < n; ++i)
        {
            m1 = a * m1 + (1 - a) * g(rng);
            m2 = a * m2 + (1 - a) * m1;
            ph += TWO_PI * dev / sr * std::clamp(m2 * 25.0, -1.0, 1.0);
            x[(size_t)i] = std::polar(1.0, ph);
        }
        return x;
    }
};

// White noise for an in-band SNR over occupied bandwidth obw.
void addNoise(std::vector<cd>& x, double snr_db, double sr, double obw, uint32_t seed)
{
    double p = 0;
    for (auto& v : x)
        p += std::norm(v);
    p /= (double)x.size();
    const double                     n_full = p / std::pow(10, snr_db / 10) * sr / obw;
    std::mt19937                     rng(seed);
    std::normal_distribution<double> g(0, std::sqrt(n_full / 2));
    for (auto& v : x)
        v += cd(g(rng), g(rng));
}

std::string classOf(const analysis::SignalFeatures& f)
{
    analysis::ModulationClassifier cls;
    analysis::AnalysisResult       r;
    cls.classify(f, r);
    if (!r.classified)
        return "unclassified: " + r.reject_reason;
    return r.digital_modulation.empty() ? r.analog_modulation : r.digital_modulation;
}

const std::vector<cd> kBpsk{{1, 0}, {-1, 0}};
const std::vector<cd> kQpsk{{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};

std::vector<cd> psk8()
{
    std::vector<cd> c;
    for (int k = 0; k < 8; ++k)
        c.push_back(std::polar(1.0, TWO_PI * k / 8));
    return c;
}

std::vector<cd> qam16()
{
    std::vector<cd> c;
    for (int i = -3; i <= 3; i += 2)
        for (int j = -3; j <= 3; j += 2)
            c.push_back(cd(i, j));
    return c;
}
}  // namespace

// ANA-08: band-limited, carrier-offset PSK/QAM — the textbook cumulants only
// hold one sample per symbol with the carrier removed; the old thresholds were
// also wrong for QPSK (C40 ≈ −2) and used the phase-dependent Re{C40}.
// Before: BPSK → QAM16, QPSK/8PSK → FM_WB.
TEST(AuditFeatures, Ana08_LinearModulationsClassified)
{
    // 30 dB in a 2 MS/s capture; 15 dB (in-band) at the rate the collector
    // requests for a 135 kHz detection (ANA-07: 1.08 MS/s).
    analysis::FeatureExtractor fx(4096);
    const double sr15 = analysis::collectSampleRate(analysis::CollectorConfig{}, 135e3);

    struct C
    {
        const char*     want;
        std::vector<cd> cons;
    } cases[] = {
        {"BPSK", kBpsk},
        {"QPSK", kQpsk},
        {"8PSK", psk8()},
        {"QAM16", qam16()},
    };

    for (double snr : {30.0, 15.0})
        for (double df : {0.0, 20e3})
            for (auto& c : cases)
            {
                const double sr = snr > 20 ? SR : sr15;
                const Gen    g{sr};
                auto         x = g.linear(c.cons, 100e3, df, 3);
                addNoise(x, snr, sr, 135e3, 5);
                auto f = fx.extract(interleave(x), au::hertz(sr), au::hertz(0.0));
                EXPECT_EQ(classOf(f), c.want) << "snr " << snr << " df " << df;
                EXPECT_TRUE(f.cumulants_symbol_spaced) << c.want;
            }
}

// ANA-08: 2-FSK came out as FM (inst. frequency spread, no PSK cumulant match).
TEST(AuditFeatures, Ana08_FskAndMsk)
{
    analysis::FeatureExtractor fx(4096);
    const double               sr15 =
        analysis::collectSampleRate(analysis::CollectorConfig{}, 100e3);  // 800 kS/s

    struct C
    {
        const char* want;
        double      h;
    } cases[] = {{"FSK", 1.0}, {"GMSK", 0.5}};

    for (double snr : {30.0, 15.0})
        for (double df : {0.0, 20e3})
            for (auto& c : cases)
            {
                const double sr = snr > 20 ? SR : sr15;
                const Gen    g{sr};
                auto         x = g.fsk(50e3, c.h, df, 4);
                addNoise(x, snr, sr, 100e3, 5);
                auto f = fx.extract(interleave(x), au::hertz(sr), au::hertz(0.0));
                EXPECT_EQ(classOf(f), c.want) << "snr " << snr << " df " << df;
                EXPECT_NEAR(f.symbol_rate_sps.in(au::hertz), 50e3, 2.5e3) << c.want;
                EXPECT_NEAR(f.fsk_mod_index, c.h, 0.15) << c.want;
            }
}

// Guard: tone-modulated analog test signals stay analog (their |x|² / x²
// lines are far below the occupied bandwidth, or there is a carrier).
TEST(AuditFeatures, Ana08_ToneAnalogStaysAnalog)
{
    analysis::FeatureExtractor fx(4096);
    EXPECT_EQ(classOf(fx.extract(fm(5e3, 1e3), au::hertz(SR), au::hertz(0.0))), "FM_NB");
    EXPECT_EQ(classOf(fx.extract(fm(75e3, 3e3), au::hertz(SR), au::hertz(0.0))), "FM_WB");
    std::vector<cd> am(NS);
    for (int i = 0; i < NS; ++i)
        am[(size_t)i] =
            (1.0 + 0.5 * std::cos(TWO_PI * 1e3 * i / SR)) * std::polar(1.0, TWO_PI * 50e3 * i / SR);
    EXPECT_EQ(classOf(fx.extract(interleave(am), au::hertz(SR), au::hertz(0.0))), "AM_DSB_LC");
}

// ANA-07: the SNR was signal / (floor × every bin of the capture): a 4.8 kBd
// QPSK at 20 dB in-band read 1.5 dB in a 2 MS/s capture → "SNR too low".
TEST(AuditFeatures, Ana07_SnrIsInBand)
{
    analysis::FeatureExtractor fx(4096);
    const Gen                  g{SR};
    auto                       x = g.linear(kQpsk, 4800, 0, 3);
    addNoise(x, 20, SR, 6480, 5);
    auto f = fx.extract(interleave(x), au::hertz(SR), au::hertz(0.0));
    EXPECT_NEAR(f.snr_db, 20.0, 3.0);
}

// ANA-07: the slow path captured at 2 MS/s whatever the detection; now at
// rate_per_bandwidth × bw — and narrowband signals classify there.
TEST(AuditFeatures, Ana07_CaptureRateFollowsBandwidth)
{
    analysis::CollectorConfig c;                                      // 2 MS/s max, ×8
    EXPECT_DOUBLE_EQ(analysis::collectSampleRate(c, 12.5e3), 200e3);  // 25 kHz floor
    EXPECT_DOUBLE_EQ(analysis::collectSampleRate(c, 100e3), 800e3);
    EXPECT_DOUBLE_EQ(analysis::collectSampleRate(c, 1e6), 2e6);
    c.rate_per_bandwidth = 0;
    EXPECT_DOUBLE_EQ(analysis::collectSampleRate(c, 12.5e3), 2e6);
}

TEST(AuditFeatures, Ana07_NarrowbandAtMatchedRate)
{
    analysis::FeatureExtractor fx(4096);
    const Gen                  g{200e3};
    auto                       q = g.linear(kQpsk, 4800, 300, 3);
    addNoise(q, 20, 200e3, 6480, 5);
    EXPECT_EQ(classOf(fx.extract(interleave(q), au::hertz(200e3), au::hertz(0.0))), "QPSK");
    auto v = g.fmVoice(2.5e3, 6);
    addNoise(v, 20, 200e3, 11e3, 7);
    EXPECT_EQ(classOf(fx.extract(interleave(v), au::hertz(200e3), au::hertz(0.0))), "FM_NB");
}

// ANA-09: −10 dB walk from the peak bin stopped at the first dip: 2-FSK
// (tones 100 kHz apart) measured 977 Hz.
TEST(AuditFeatures, Ana09_OccupiedBandwidthOfFsk)
{
    analysis::FeatureExtractor fx(4096);
    const Gen                  g{SR};
    auto                       x = g.fsk(50e3, 1.0, 0, 4);
    addNoise(x, 30, SR, 100e3, 5);
    auto f = fx.extract(interleave(x), au::hertz(SR), au::hertz(0.0));
    EXPECT_GT(f.bandwidth_hz.in(au::hertz), 80e3);
    EXPECT_LT(f.bandwidth_hz.in(au::hertz), 160e3);
}

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
