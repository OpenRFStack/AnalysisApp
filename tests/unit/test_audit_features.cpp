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
#include <au/units/hertz.hh>
#include <gtest/gtest.h>
#include "FeatureExtractor.hpp"
#include "ModulationClassifier.hpp"
#include "AnalysisResult.hpp"
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>

namespace {
constexpr double TWO_PI = 2.0 * std::numbers::pi_v<double>;
constexpr double SR     = 2'000'000.0;
constexpr int    NS     = 65536;
using cd = std::complex<double>;

std::vector<float> interleave(const std::vector<cd>& x) {
    std::vector<float> iq;
    iq.reserve(x.size() * 2);
    for (auto& s : x) { iq.push_back((float)s.real()); iq.push_back((float)s.imag()); }
    return iq;
}

// OFDM with QPSK subcarriers and a cyclic prefix, starting `offset` samples
// into a symbol — a capture never starts on a symbol boundary.
std::vector<float> ofdm(int nfft, int ncp, int offset, uint32_t seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, 3);
    const cd q[4] = {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
    std::vector<cd> out;
    while ((int)out.size() < NS + offset) {
        std::vector<cd> td(nfft);
        std::vector<cd> fd(nfft);
        for (int k = 1; k < nfft; ++k) fd[k] = q[d(rng)];
        for (int n = 0; n < nfft; ++n) {
            cd acc{0, 0};
            for (int k = 0; k < nfft; ++k) acc += fd[k] * std::polar(1.0, TWO_PI * k * n / nfft);
            td[n] = acc / (double)nfft;
        }
        for (int k = nfft - ncp; k < nfft; ++k) out.push_back(td[k]);
        for (int k = 0; k < nfft; ++k) out.push_back(td[k]);
    }
    return interleave({out.begin() + offset, out.begin() + offset + NS});
}

std::vector<float> tone(double f) {
    std::vector<cd> x(NS);
    for (int i = 0; i < NS; ++i) x[i] = std::polar(1.0, TWO_PI * f * i / SR);
    return interleave(x);
}

std::vector<float> fm(double dev, double fmod) {
    std::vector<cd> x(NS);
    double ph = 0;
    for (int i = 0; i < NS; ++i) {
        ph += TWO_PI * dev / SR * std::cos(TWO_PI * fmod * i / SR);
        x[i] = std::polar(1.0, ph);
    }
    return interleave(x);
}

// Band-limited (raised-cosine, β = 0.35) PSK at symbol rate rs, carrier offset df.
std::vector<float> psk(int m, double rs, double df, uint32_t seed = 3) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, m - 1);
    const double sps = SR / rs;
    const int nsym = (int)(NS / sps) + 40;
    std::vector<cd> sym(nsym);
    for (auto& s : sym) s = std::polar(1.0, TWO_PI * d(rng) / m + (m == 4 ? TWO_PI / 8 : 0.0));
    auto rc = [&](double t) {           // t in symbols
        const double b = 0.35;
        if (std::abs(t) < 1e-9) return 1.0;
        if (std::abs(std::abs(2 * b * t) - 1.0) < 1e-9)
            return std::numbers::pi / 4 * std::sin(std::numbers::pi * t) / (std::numbers::pi * t);
        return std::sin(std::numbers::pi * t) / (std::numbers::pi * t) *
               std::cos(std::numbers::pi * b * t) / (1 - 4 * b * b * t * t);
    };
    std::vector<cd> x(NS);
    for (int i = 0; i < NS; ++i) {
        const double t = i / sps + 20;  // skip the filter warm-up
        const int k0 = (int)t;
        cd acc{0, 0};
        for (int k = k0 - 12; k <= k0 + 12; ++k)
            if (k >= 0 && k < nsym) acc += sym[k] * rc(t - k);
        x[i] = acc * std::polar(1.0, TWO_PI * df * i / SR);
    }
    return interleave(x);
}
} // namespace

// ANA-03: the cyclic-prefix correlator summed windows at n = 0, L, 2L …, i.e.
// it assumed the capture starts exactly on an OFDM symbol boundary.
TEST(AuditFeatures, Ana03_OfdmDetectedAtAnyTimingOffset) {
    analysis::FeatureExtractor fx(4096);
    for (int off : {0, 7, 29, 41, 63}) {
        auto f = fx.extract(ofdm(64, 16, off), au::hertz(SR), au::hertz(0.0));
        EXPECT_TRUE(f.ofdm_detected) << "offset " << off;
        if (f.ofdm_detected) {
            EXPECT_EQ(f.ofdm_fft_size_est, 64) << "offset " << off;
            EXPECT_NEAR(f.ofdm_cp_ratio, 0.25, 1e-9) << "offset " << off;
        }
    }
}

// ANA-04: a narrowband signal is (nearly) periodic at every lag, so the CP
// score was ~1 for any Nfft — a CW carrier or an FM voice channel came out as
// "OFDM" (OFDM is decided before the analog/digital split).
TEST(AuditFeatures, Ana04_NarrowbandSignalsAreNotOfdm) {
    analysis::FeatureExtractor fx(4096);
    analysis::ModulationClassifier cls;
    struct Case { const char* name; std::vector<float> iq; } cases[] = {
        {"CW tone",      tone(100e3)},
        {"FM 5k/1k",     fm(5e3, 1e3)},
        {"FM 75k/3k",    fm(75e3, 3e3)},
    };
    for (auto& c : cases) {
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
TEST(AuditFeatures, Ana05_SymbolRateOfBandLimitedPsk) {
    analysis::FeatureExtractor fx(4096);
    struct Case { int m; double rs, df; } cases[] = {
        {2, 100e3, 0.0}, {4, 100e3, 0.0}, {4, 100e3, 20e3}, {2, 50e3, -15e3}, {4, 250e3, 5e3},
    };
    for (auto& c : cases) {
        auto f = fx.extract(psk(c.m, c.rs, c.df), au::hertz(SR), au::hertz(0.0));
        EXPECT_NEAR(f.symbol_rate_sps.in(au::hertz), c.rs, c.rs * 0.05)
            << c.m << "-PSK Rs=" << c.rs << " df=" << c.df;
    }
}

TEST(AuditFeatures, Ana05_SymbolRateAt10dBSnr) {
    analysis::FeatureExtractor fx(4096);
    auto iq = psk(4, 100e3, 12e3, 9);
    double p = 0;
    for (float v : iq) p += (double)v * v;
    p /= (iq.size() / 2);
    std::mt19937 rng(2);
    std::normal_distribution<float> g(0.f, (float)std::sqrt(p / 10.0 / 2.0));
    for (auto& v : iq) v += g(rng);
    auto f = fx.extract(iq, au::hertz(SR), au::hertz(0.0));
    EXPECT_NEAR(f.symbol_rate_sps.in(au::hertz), 100e3, 5e3);
}

// Guard: no line, no symbol rate (noise, CW, wideband FM).
TEST(AuditFeatures, Ana05_NoSymbolRateWithoutALine) {
    analysis::FeatureExtractor fx(4096);
    std::mt19937 rng(1);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> noise(2 * NS);
    for (auto& v : noise) v = g(rng);
    EXPECT_EQ(fx.extract(noise, au::hertz(SR), au::hertz(0.0)).symbol_rate_sps.in(au::hertz), 0.0);
    EXPECT_EQ(fx.extract(tone(100e3), au::hertz(SR), au::hertz(0.0)).symbol_rate_sps.in(au::hertz), 0.0);
}

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
