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
#include "FeatureExtractor.hpp"
#include <au/units/hertz.hh>
#include <au/units/seconds.hh>
#include <fftw3.h>
#include <numbers>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <cstring>
#include <spdlog/spdlog.h>

namespace analysis {

using cf32 = std::complex<float>;

// ── Constructor / Destructor ──────────────────────────────────────────────────

FeatureExtractor::FeatureExtractor(int fft_size)
    : fft_size_(fft_size)
{
    // Allocate persistent FFTW plans for the primary FFT size
    auto* in1  = reinterpret_cast<fftwf_complex*>(
                     fftwf_malloc(sizeof(fftwf_complex) * fft_size_));
    auto* out1 = reinterpret_cast<fftwf_complex*>(
                     fftwf_malloc(sizeof(fftwf_complex) * fft_size_));
    fft_plan_ = fftwf_plan_dft_1d(fft_size_, in1, out1,
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_free(in1);
    fftwf_free(out1);

    // Second plan for cyclostationary / symbol-rate FFT (same size)
    auto* in2  = reinterpret_cast<fftwf_complex*>(
                     fftwf_malloc(sizeof(fftwf_complex) * fft_size_));
    auto* out2 = reinterpret_cast<fftwf_complex*>(
                     fftwf_malloc(sizeof(fftwf_complex) * fft_size_));
    fft_plan2_ = fftwf_plan_dft_1d(fft_size_, in2, out2,
                                    FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_free(in2);
    fftwf_free(out2);

    hann_.resize((size_t)fft_size_);
    for (int i = 0; i < fft_size_; ++i)
        hann_[(size_t)i] = static_cast<float>(
            0.5 * (1.0 - std::cos(2.0 * std::numbers::pi_v<double> * i / (fft_size_ - 1))));

    for (int n = fft_size_; n <= std::max(fft_size_, kMaxSymbolRateLen); n *= 2) {
        auto* a = fftwf_alloc_complex((size_t)n);
        auto* b = fftwf_alloc_complex((size_t)n);
        sym_plans_.push_back(fftwf_plan_dft_1d(n, a, b, FFTW_FORWARD, FFTW_ESTIMATE));
        std::vector<float> w((size_t)n);
        for (int i = 0; i < n; ++i)
            w[(size_t)i] = static_cast<float>(
                0.5 * (1.0 - std::cos(2.0 * std::numbers::pi_v<double> * i / (n - 1))));
        sym_windows_.push_back(std::move(w));
        fftwf_free(a);
        fftwf_free(b);
    }
}

FeatureExtractor::~FeatureExtractor()
{
    if (fft_plan_)  fftwf_destroy_plan(reinterpret_cast<fftwf_plan>(fft_plan_));
    if (fft_plan2_) fftwf_destroy_plan(reinterpret_cast<fftwf_plan>(fft_plan2_));
    for (void* p : sym_plans_)
        if (p) fftwf_destroy_plan(reinterpret_cast<fftwf_plan>(p));
}

int FeatureExtractor::symPlanIndex(int n) const
{
    int idx = 0;
    for (int m = fft_size_; m < n; m *= 2) ++idx;
    return idx < (int)sym_plans_.size() ? idx : -1;
}

// ── Utility helpers ───────────────────────────────────────────────────────────

static std::vector<cf32> toComplex(const std::vector<float>& iq)
{
    std::vector<cf32> out;
    out.reserve(iq.size() / 2);
    for (size_t i = 0; i + 1 < iq.size(); i += 2)
        out.emplace_back(iq[i], iq[i+1]);
    return out;
}

// FFT a block of complex<float> samples; result has fft_size bins.
// Aligned scratch is per thread and reused (it used to be two fftwf_malloc /
// fftwf_free per transform); extract() runs on several workers concurrently.
namespace {
struct FftScratch {
    fftwf_complex* in  = nullptr;
    fftwf_complex* out = nullptr;
    int cap = 0;
    ~FftScratch() { if (in) fftwf_free(in); if (out) fftwf_free(out); }
    void ensure(int n) {
        if (n <= cap) return;
        if (in)  fftwf_free(in);
        if (out) fftwf_free(out);
        in  = fftwf_alloc_complex((size_t)n);
        out = fftwf_alloc_complex((size_t)n);
        cap = n;
    }
};
} // namespace

static void runFftPlan(fftwf_plan plan,
                       const cf32* in, int fft_size,
                       std::vector<cf32>& out)
{
    thread_local FftScratch sc;
    sc.ensure(fft_size);
    std::memcpy(sc.in, in, sizeof(fftwf_complex) * (size_t)fft_size);
    fftwf_execute_dft(plan, sc.in, sc.out);
    out.resize((size_t)fft_size);
    std::memcpy(out.data(), sc.out, sizeof(fftwf_complex) * (size_t)fft_size);
}

// ── PSD (Welch method, 8 frames, 50% overlap, Hann window) ───────────────────

void FeatureExtractor::computePsd(const std::vector<cf32>& x,
                                   std::vector<float>& psd_out) const
{
    const int N   = fft_size_;
    const int hop = N / 2;                       // 50% overlap
    const int frames = 8;
    const auto& hann = hann_;

    psd_out.assign(N, 0.0f);

    // Remove DC offset (LO leakage) over the full block before framing.
    cf32 block_mean{0.f, 0.f};
    int  block_len = std::min(static_cast<int>(x.size()), frames * hop + N);
    for (int i = 0; i < block_len; ++i) block_mean += x[i];
    block_mean /= static_cast<float>(block_len);

    int frame_count = 0;
    std::vector<cf32> windowed((size_t)N), spec;
    for (int f = 0; f < frames; ++f) {
        int offset = f * hop;
        if (offset + N > static_cast<int>(x.size())) break;

        for (int i = 0; i < N; ++i)
            windowed[i] = (x[offset + i] - block_mean) * hann[i];

        runFftPlan(reinterpret_cast<fftwf_plan>(fft_plan_),
                   windowed.data(), N, spec);

        for (int i = 0; i < N; ++i)
            psd_out[i] += std::norm(spec[i]);

        ++frame_count;
    }

    if (frame_count == 0) return;

    // Normalize: divide by frame count and window power, then scale so bins
    // sum to total power.
    float win_power = 0;
    for (float w : hann) win_power += w * w;

    float total = 0;
    for (int i = 0; i < N; ++i) {
        psd_out[i] /= (frame_count * win_power);
        total += psd_out[i];
    }

    // Rearrange to DC-centered (fftshift)
    std::rotate(psd_out.begin(), psd_out.begin() + N/2, psd_out.end());
}

// ── Cumulants ─────────────────────────────────────────────────────────────────

void FeatureExtractor::computeCumulants(const std::vector<cf32>& x,
                                         double& c40r, double& c40i,
                                         double& c41r, double& c42) const
{
    const int N = static_cast<int>(x.size());
    if (N == 0) { c40r = c40i = c41r = c42 = 0; return; }

    // Moments
    std::complex<double> M20{0,0};
    double               M21 = 0;
    std::complex<double> M40{0,0};
    std::complex<double> M41{0,0};
    double               M42 = 0;

    for (const auto& s : x) {
        std::complex<double> sd(s.real(), s.imag());
        double mag2 = std::norm(sd);

        M20 += sd * sd;
        M21 += mag2;
        M40 += sd * sd * sd * sd;
        M41 += sd * sd * sd * std::conj(sd);
        M42 += mag2 * mag2;
    }

    M20 /= N;
    M21 /= N;
    M40 /= N;
    M41 /= N;
    M42 /= N;

    double M21_sq = M21 * M21;
    if (M21_sq < 1e-30) { c40r = c40i = c41r = c42 = 0; return; }

    // Cumulants
    std::complex<double> C40 = M40 - 3.0 * M20 * M20;
    std::complex<double> C41 = M41 - 3.0 * M20 * M21;
    double               C42 = M42 - std::norm(M20) - 2.0 * M21_sq;

    c40r = C40.real() / M21_sq;
    c40i = C40.imag() / M21_sq;
    c41r = C41.real() / M21_sq;
    c42  = C42        / M21_sq;
}

// ── Symbol-rate estimation ────────────────────────────────────────────────────
//
// Three cyclostationary line sources, each searched for the strongest
// *narrow* line (contrast against the median of its 64-bin neighbourhood —
// data self-noise is a broad hump that out-ranked the line against a global
// mean):
//   · |x|²  — line at Rs (any linear modulation with excess bandwidth)
//   · x²    — carrier at 2Δf, lines at 2Δf ± Rs (BPSK)
//   · x⁴    — carrier at 4Δf, lines at 4Δf ± Rs (QPSK / M-PSK)
// For x² / x⁴ the symbol rate is the distance from the carrier line (ANA-05:
// the x⁴ line *position* / 4 was returned — Rs/4 for a centred signal, the
// carrier offset for an offset one). The FFT spans the whole block up to
// kMaxSymbolRateLen samples: 4096 samples (~200 symbols) left the line below
// the self-noise peaks.

namespace {
struct Line { double contrast = 0.0; int bin = 0; };

// Strongest narrow line in v[kMinBin .. v.size()), by v[d] / median(64-bin block).
constexpr int kMinBin = 8;   // clear of the (windowed) carrier / DC main lobe
Line strongestLine(const std::vector<float>& v)
{
    constexpr int B = 64;
    const int n = static_cast<int>(v.size());
    Line best;
    std::vector<float> blk;
    blk.reserve(B);
    for (int b0 = 0; b0 < n; b0 += B) {
        const int b1 = std::min(n, b0 + B);
        blk.assign(v.begin() + b0, v.begin() + b1);
        auto mid = blk.begin() + (ptrdiff_t)blk.size() / 2;
        std::nth_element(blk.begin(), mid, blk.end());
        const double med = std::max(1e-30, (double)*mid);
        for (int d = std::max(b0, kMinBin); d < b1; ++d) {
            const double c = v[(size_t)d] / med;
            if (c > best.contrast) best = {c, d};
        }
    }
    return best;
}
} // namespace

double FeatureExtractor::estimateSymbolRate(const std::vector<cf32>& x,
                                             au::QuantityD<au::Hertz> sample_rate_sps) const
{
    const double sample_rate = sample_rate_sps.in(au::hertz);
    int L = fft_size_;
    while (L * 2 <= std::min<int>((int)x.size(), kMaxSymbolRateLen)) L *= 2;
    if (static_cast<int>(x.size()) < L) return 0.0;
    const int pi = symPlanIndex(L);
    if (pi < 0) return 0.0;
    auto plan = reinterpret_cast<fftwf_plan>(sym_plans_[(size_t)pi]);
    const auto& win = sym_windows_[(size_t)pi];   // Hann: carrier leakage ∝ 1/d³

    std::vector<cf32> y((size_t)L), spec;
    std::vector<float> mag((size_t)L), side((size_t)L / 2);

    // Total power guards against ratios of rounding noise (pure tone: the
    // |x|² spectrum is ~0 everywhere).
    double p_mean = 0.0;
    for (int i = 0; i < L; ++i) p_mean += std::norm(x[i]);
    p_mean /= L;
    if (p_mean < 1e-20) return 0.0;

    Line best;
    // |x|²
    for (int i = 0; i < L; ++i) y[i] = cf32((float)(std::norm(x[i]) - p_mean) * win[i], 0.0f);
    runFftPlan(plan, y.data(), L, spec);
    for (int d = 0; d < L / 2; ++d) side[d] = std::sqrt(std::norm(spec[d]));
    {
        Line l = strongestLine(side);
        // A real line carries a meaningful share of the envelope power.
        if (side[(size_t)l.bin] > 1e-4 * p_mean * L) best = l;
    }

    // x² and x⁴: distance of the strongest sideline pair from the carrier line
    for (int p : {2, 4}) {
        for (int i = 0; i < L; ++i) {           // explicit: avoids __mulsc3
            const float re = x[i].real(), im = x[i].imag();
            float r2 = re * re - im * im, i2 = 2.f * re * im;
            if (p == 4) { const float t = r2 * r2 - i2 * i2; i2 = 2.f * r2 * i2; r2 = t; }
            y[i] = cf32(r2 * win[i], i2 * win[i]);
        }
        runFftPlan(plan, y.data(), L, spec);
        int k_c = 0;
        for (int k = 0; k < L; ++k) {
            mag[k] = std::sqrt(std::norm(spec[k]));   // not cabsf/hypotf
            if (mag[k] > mag[k_c]) k_c = k;
        }
        for (int d = 0; d < L / 2; ++d)
            side[d] = 0.5f * (mag[(size_t)((k_c + d) % L)] + mag[(size_t)((k_c - d + L) % L)]);
        Line l = strongestLine(side);
        if (side[(size_t)l.bin] > 1e-4 * mag[(size_t)k_c] && l.contrast > best.contrast) best = l;
    }

    constexpr double kMinContrast = 8.0;
    if (best.contrast < kMinContrast || best.bin <= 0) return 0.0;
    return (double)best.bin * sample_rate / L;
}

// ── OFDM detection (cyclic-prefix correlation) ───────────────────────────────
//
// For every candidate (Nfft, Ncp) the lag-Nfft products p[n] = x[n]·x*[n+Nfft]
// are summed over CP-length windows spaced one symbol (L = Nfft+Ncp) apart,
// for every timing residue r ∈ [0, L) — a capture does not start on a symbol
// boundary (ANA-03: the windows used to be anchored at n = 0). The products
// are folded by n mod L in one sequential pass and the CP window is a circular
// moving sum over the fold, so each candidate costs O(samples + L).
//
// A CP makes the lag-Nfft correlation *periodic in symbol timing*: ~1 inside
// the CP, ~0 elsewhere. A narrowband signal (CW, FM voice, an oversampled
// carrier) correlates at every lag and every timing, which used to score ~1
// and come out as "OFDM" (ANA-04). The detector therefore also requires the
// correlation over *all* samples to be well below the CP-window peak.

bool FeatureExtractor::detectOfdm(const std::vector<cf32>& x,
                                   au::QuantityD<au::Hertz> /*sample_rate_sps*/,
                                   int& fft_size_out,
                                   double& cp_ratio_out) const
{
    static const int    nffts[]    = {64, 128, 256, 512, 1024, 2048};
    static const double cp_fracs[] = {1.0/4, 1.0/8, 1.0/16};
    constexpr double kMinScore    = 0.7;   // CP-window correlation at the best timing
    constexpr double kMaxContrast = 0.5;   // all-sample correlation / CP-window peak

    const int total = static_cast<int>(x.size());
    double best_score = 0.0;
    int    best_nfft  = 0;
    double best_cp    = 0.0;

    // Structure-of-arrays with explicit real arithmetic: std::complex
    // operator* goes through __mulsc3 (NaN/Inf rules) and dominated the cost.
    const float* xs = reinterpret_cast<const float*>(x.data());
    std::vector<float>  pr, pim, ea, eb;        // lag products, |x[n]|², |x[n+Nfft]|²
    std::vector<float>  fr, fim, fa, fb;        // folds by n mod L (≤ ~1000 terms each)

    for (int nfft : nffts) {
        const int M = total - nfft;
        if (M <= 0) continue;
        pr.resize((size_t)M); pim.resize((size_t)M); ea.resize((size_t)M); eb.resize((size_t)M);
        double cr_all = 0.0, ci_all = 0.0, a_all = 0.0, b_all = 0.0;
        for (int n = 0; n < M; ++n) {
            const float ar = xs[2*n], ai = xs[2*n+1];
            const float br = xs[2*(n+nfft)], bi = xs[2*(n+nfft)+1];
            pr[n]  = ar * br + ai * bi;           // a · conj(b)
            pim[n] = ai * br - ar * bi;
            ea[n]  = ar * ar + ai * ai;
            eb[n]  = br * br + bi * bi;
            cr_all += pr[n]; ci_all += pim[n]; a_all += ea[n]; b_all += eb[n];
        }
        if (a_all < 1e-12 || b_all < 1e-12) continue;
        const double rho_all = std::hypot(cr_all, ci_all) / std::sqrt(a_all * b_all);
        if (rho_all > kMaxContrast) continue;   // peak ≤ 1: cannot pass the contrast test

        for (double cp_frac : cp_fracs) {
            const int ncp = static_cast<int>(std::round(nfft * cp_frac));
            if (ncp < 1) continue;
            const int L = nfft + ncp;
            if (L * 4 > total) continue;   // need several symbols

            fr.assign((size_t)L, 0.f); fim.assign((size_t)L, 0.f);
            fa.assign((size_t)L, 0.f); fb.assign((size_t)L, 0.f);
            for (int n0 = 0; n0 < M; n0 += L) {
                const int len = std::min(L, M - n0);
                for (int j = 0; j < len; ++j) {
                    fr[j]  += pr[n0 + j];  fim[j] += pim[n0 + j];
                    fa[j]  += ea[n0 + j];  fb[j]  += eb[n0 + j];
                }
            }
            // Circular moving sum of width ncp: window starting at residue r.
            double wr = 0.0, wi = 0.0, wa = 0.0, wb = 0.0;
            for (int k = 0; k < ncp; ++k) { wr += fr[k]; wi += fim[k]; wa += fa[k]; wb += fb[k]; }
            double peak2 = 0.0;                    // max |c|² / (pa·pb)
            for (int r = 0; r < L; ++r) {
                const double den = wa * wb;
                if (den > 1e-24) peak2 = std::max(peak2, (wr * wr + wi * wi) / den);
                const int in = (r + ncp) % L;
                wr += fr[in] - fr[r]; wi += fim[in] - fim[r];
                wa += fa[in] - fa[r]; wb += fb[in] - fb[r];
            }
            const double peak = std::sqrt(peak2);
            if (peak < kMinScore || rho_all > kMaxContrast * peak) continue;
            if (peak > best_score) {
                best_score = peak;
                best_nfft  = nfft;
                best_cp    = cp_frac;
            }
        }
    }

    if (best_nfft == 0) return false;
    fft_size_out = best_nfft;
    cp_ratio_out = best_cp;
    spdlog::debug("OFDM detected: score={:.3f} Nfft={} cp_ratio={:.3f}",
                  best_score, fft_size_out, cp_ratio_out);
    return true;
}

// ── FHSS detection ────────────────────────────────────────────────────────────

bool FeatureExtractor::detectFhss(const std::vector<cf32>& x,
                                   au::QuantityD<au::Hertz> sample_rate_sps,
                                   au::QuantityD<au::Hertz>& hop_rate_out) const
{
    const double sr = sample_rate_sps.in(au::hertz);
    const int total = static_cast<int>(x.size());
    const int num_segs = 32;
    const int seg_len  = total / num_segs;
    if (seg_len < fft_size_) return false;

    // For each segment compute spectral centroid
    std::vector<double> centroids(num_segs);
    const int n = fft_size_;            // seg_len >= fft_size_ (checked above)
    std::vector<cf32> block((size_t)n), spec;
    for (int s = 0; s < num_segs; ++s) {
        int offset = s * seg_len;
        for (int i = 0; i < n; ++i) block[i] = x[offset + i] * hann_[i];

        runFftPlan(reinterpret_cast<fftwf_plan>(fft_plan_),
                   block.data(), n, spec);

        double num = 0, den = 0;
        for (int k = 0; k < n; ++k) {
            double f  = (double)k / n * sr - sr / 2.0;
            double pw = std::norm(spec[(k + n/2) % n]);
            num += f * pw;
            den += pw;
        }
        centroids[s] = (den > 1e-20) ? num / den : 0.0;
    }

    // Compute std of centroids
    double mean_c = std::accumulate(centroids.begin(), centroids.end(), 0.0) / num_segs;
    double var_c  = 0;
    for (double c : centroids) var_c += (c - mean_c) * (c - mean_c);
    var_c /= num_segs;
    double std_c = std::sqrt(var_c);

    // Bandwidth estimate from signal itself
    double sig_bw = sr * 0.5; // conservative

    if (std_c > sig_bw / 4.0) {
        // Estimate hop rate from transitions between segments
        int transitions = 0;
        for (int s = 1; s < num_segs; ++s)
            if (std::abs(centroids[s] - centroids[s-1]) > std_c * 0.5)
                ++transitions;

        double total_time_s = (double)total / sr;
        hop_rate_out = au::hertz((double)transitions / total_time_s);
        return true;
    }
    return false;
}

// ── Chirp detection ───────────────────────────────────────────────────────────

bool FeatureExtractor::detectChirp(const std::vector<cf32>& x,
                                    au::QuantityD<au::Hertz> sample_rate_sps,
                                    double& chirp_rate_out) const
{
    const double sr = sample_rate_sps.in(au::hertz);
    const int N = std::min(1000, static_cast<int>(x.size()) - 1);
    if (N < 10) return false;

    // Instantaneous frequency
    std::vector<double> fi(N);
    for (int i = 0; i < N; ++i) {
        cf32 prod = x[i + 1] * std::conj(x[i]);
        fi[i] = std::arg(prod) * sr / (2.0 * std::numbers::pi_v<double>);
    }

    // Linear regression on fi vs sample index
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (int i = 0; i < N; ++i) {
        sum_x  += i;
        sum_y  += fi[i];
        sum_xx += (double)i * i;
        sum_xy += (double)i * fi[i];
    }
    double denom = N * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-12) return false;

    double slope     = (N * sum_xy - sum_x * sum_y) / denom;
    double intercept = (sum_y - slope * sum_x) / N;

    // Compute R^2
    double ss_res = 0, ss_tot = 0;
    double mean_y = sum_y / N;
    for (int i = 0; i < N; ++i) {
        double pred = slope * i + intercept;
        ss_res += (fi[i] - pred) * (fi[i] - pred);
        ss_tot += (fi[i] - mean_y) * (fi[i] - mean_y);
    }
    double r2 = (ss_tot > 1e-12) ? 1.0 - ss_res / ss_tot : 0.0;

    // Slope in Hz/sample → convert to Hz/s
    double slope_hz_s = slope * sr;

    if (r2 > 0.95 && std::abs(slope_hz_s) > sr / 100.0) {
        chirp_rate_out = slope_hz_s;
        return true;
    }
    return false;
}

// ── Instantaneous frequency statistics ───────────────────────────────────────

void FeatureExtractor::computeInstFreqStats(const std::vector<cf32>& x,
                                             au::QuantityD<au::Hertz> sample_rate_sps,
                                             au::QuantityD<au::Hertz>& mean,
                                             au::QuantityD<au::Hertz>& std_dev) const
{
    const double sr = sample_rate_sps.in(au::hertz);
    const int N = static_cast<int>(x.size()) - 1;
    if (N < 2) { mean = au::hertz(0.0); std_dev = au::hertz(0.0); return; }

    std::vector<double> fi(N);
    for (int i = 0; i < N; ++i) {
        cf32 prod = x[i + 1] * std::conj(x[i]);
        fi[i] = std::arg(prod) * sr / (2.0 * std::numbers::pi_v<double>);
    }

    double sum = std::accumulate(fi.begin(), fi.end(), 0.0);
    double mean_hz = sum / N;
    mean = au::hertz(mean_hz);

    double var = 0;
    for (double f : fi) var += (f - mean_hz) * (f - mean_hz);
    std_dev = au::hertz(std::sqrt(var / N));
}

// ── Burst detection ───────────────────────────────────────────────────────────

bool FeatureExtractor::detectBurst(const std::vector<cf32>& x,
                                    double& duty_cycle,
                                    double& period_samples) const
{
    const int total    = static_cast<int>(x.size());
    const int windows  = 100;
    const int win_size = total / windows;
    if (win_size < 1) { duty_cycle = 1.0; period_samples = 0; return false; }

    std::vector<double> power(windows);
    double peak_pwr = 0;
    for (int w = 0; w < windows; ++w) {
        double p = 0;
        for (int i = w * win_size; i < (w + 1) * win_size; ++i)
            p += std::norm(x[i]);
        power[w] = p / win_size;
        if (power[w] > peak_pwr) peak_pwr = power[w];
    }

    const double threshold = peak_pwr * 0.1;
    int on_count = 0;
    std::vector<int> on_off(windows);
    for (int w = 0; w < windows; ++w) {
        on_off[w] = (power[w] > threshold) ? 1 : 0;
        on_count += on_off[w];
    }

    duty_cycle = (double)on_count / windows;

    if (duty_cycle >= 0.85) {
        period_samples = 0;
        return false;
    }

    // Estimate period via autocorrelation of on/off sequence
    int best_lag = 0;
    double best_ac = -1e9;
    for (int lag = 2; lag < windows / 2; ++lag) {
        double ac = 0;
        for (int w = 0; w < windows - lag; ++w)
            ac += on_off[w] * on_off[w + lag];
        if (ac > best_ac) {
            best_ac  = ac;
            best_lag = lag;
        }
    }

    period_samples = (best_lag > 0)
                     ? (double)best_lag * win_size
                     : 0.0;
    return true;
}

// ── Main extraction entry ─────────────────────────────────────────────────────

SignalFeatures FeatureExtractor::extract(const std::vector<float>& iq,
                                          au::QuantityD<au::Hertz> sample_rate_sps,
                                          au::QuantityD<au::Hertz> center_freq_hz) const
{
    const double sample_rate = sample_rate_sps.in(au::hertz);
    SignalFeatures f;
    f.center_freq_hz  = center_freq_hz;
    f.sample_rate_sps = sample_rate_sps;
    f.sample_count    = static_cast<int>(iq.size() / 2);

    if (f.sample_count < fft_size_) {
        spdlog::warn("FeatureExtractor: too few samples ({} < {})",
                     f.sample_count, fft_size_);
        return f;
    }

    auto x = toComplex(iq);
    const int N_samp = static_cast<int>(x.size());

    // ── PSD ──────────────────────────────────────────────────────────────
    std::vector<float> psd;
    computePsd(x, psd);

    const int N_psd = static_cast<int>(psd.size());
    double total_power = 0;
    for (float p : psd) total_power += p;

    // Peak bin and noise floor (bottom 20%)
    float peak_psd = *std::max_element(psd.begin(), psd.end());

    std::vector<float> sorted_psd(psd);
    std::sort(sorted_psd.begin(), sorted_psd.end());
    int noise_bins = std::max(1, N_psd / 5);
    double noise_sum = 0;
    for (int i = 0; i < noise_bins; ++i) noise_sum += sorted_psd[i];
    double noise_floor = noise_sum / noise_bins;
    double noise_power = noise_floor * N_psd;

    double signal_power = total_power - noise_power;
    if (signal_power < 0) signal_power = 0;

    // SNR
    if (noise_power > 1e-30)
        f.snr_db = 10.0 * std::log10(signal_power / noise_power + 1e-20);
    else
        f.snr_db = 60.0;

    // Occupied bandwidth at -10 dB from peak
    float threshold = peak_psd / 10.0f;   // -10 dB
    int peak_bin = static_cast<int>(
        std::max_element(psd.begin(), psd.end()) - psd.begin());

    int lo = peak_bin, hi = peak_bin;
    while (lo > 0         && psd[lo - 1] > threshold) --lo;
    while (hi < N_psd - 1 && psd[hi + 1] > threshold) ++hi;

    f.bandwidth_hz = au::hertz((double)(hi - lo) * sample_rate / N_psd);

    // Spectral flatness in signal band (geometric mean / arithmetic mean)
    {
        double log_sum = 0, lin_sum = 0;
        int band_bins = hi - lo + 1;
        for (int k = lo; k <= hi; ++k) {
            float p = std::max(psd[k], 1e-20f);
            log_sum += std::log(p);
            lin_sum += p;
        }
        double geom = std::exp(log_sum / band_bins);
        double arith = lin_sum / band_bins;
        f.spectral_flatness = (arith > 1e-30) ? geom / arith : 0.0;
    }

    // Spectral symmetry (correlation of left/right halves around center)
    {
        int half = (hi - lo) / 2;
        if (half > 0) {
            int center = (lo + hi) / 2;
            double cov = 0, var_l = 0, var_r = 0;
            double mean_l = 0, mean_r = 0;
            for (int k = 0; k < half; ++k) {
                mean_l += psd[center - k - 1];
                mean_r += psd[center + k + 1];
            }
            mean_l /= half; mean_r /= half;
            for (int k = 0; k < half; ++k) {
                double dl = psd[center - k - 1] - mean_l;
                double dr = psd[center + k + 1] - mean_r;
                cov   += dl * dr;
                var_l += dl * dl;
                var_r += dr * dr;
            }
            double denom = std::sqrt(var_l * var_r);
            f.spectral_symmetry = (denom > 1e-20) ? cov / denom : 0.0;
            f.spectral_symmetry = std::max(-1.0, std::min(1.0, f.spectral_symmetry));
        } else {
            f.spectral_symmetry = 1.0;
        }
    }

    // ── Envelope statistics ───────────────────────────────────────────────
    {
        std::vector<float> env(N_samp);
        float env_min = 1e30f, env_max = 0.0f;
        double env_sum = 0;
        for (int i = 0; i < N_samp; ++i) {
            float e = std::abs(x[i]);
            env[i]   = e;
            env_sum += e;
            if (e < env_min) env_min = e;
            if (e > env_max) env_max = e;
        }
        f.envelope_mean = env_sum / N_samp;

        double var = 0;
        for (float e : env) var += ((double)e - f.envelope_mean) * ((double)e - f.envelope_mean);
        f.envelope_std = std::sqrt(var / N_samp);

        double mean2 = f.envelope_mean * f.envelope_mean;
        f.envelope_variance_norm = (mean2 > 1e-30)
                                   ? (var / N_samp) / mean2
                                   : 0.0;

        // AM index
        float sum_env_fb = env_max + env_min;
        f.am_index = (sum_env_fb > 1e-12f)
                     ? (double)(env_max - env_min) / sum_env_fb
                     : 0.0;
    }

    // ── Instantaneous frequency stats ─────────────────────────────────────
    computeInstFreqStats(x, sample_rate_sps,
                         f.inst_freq_mean_hz, f.inst_freq_std_hz);

    // Instantaneous phase std
    {
        // Phase of each sample
        double prev_phase = 0, phase_sum = 0;
        std::vector<double> phases(N_samp);
        for (int i = 0; i < N_samp; ++i) {
            phases[i] = std::arg(x[i]);
            phase_sum += phases[i];
        }
        double mean_ph = phase_sum / N_samp;
        double var_ph = 0;
        for (double ph : phases) var_ph += (ph - mean_ph) * (ph - mean_ph);
        f.inst_phase_std = std::sqrt(var_ph / N_samp);
        (void)prev_phase;
    }

    // FM deviation estimate
    f.fm_deviation_hz = f.inst_freq_std_hz;

    // ── Higher-order cumulants ────────────────────────────────────────────
    computeCumulants(x, f.c40_real, f.c40_imag, f.c41_real, f.c42);

    // ── Symbol rate ───────────────────────────────────────────────────────
    f.symbol_rate_sps = au::hertz(estimateSymbolRate(x, sample_rate_sps));

    // ── Structural features ───────────────────────────────────────────────
    double period_samp = 0;
    f.is_burst = detectBurst(x, f.burst_duty_cycle, period_samp);
    if (period_samp > 0)
        f.burst_period_ms = au::seconds(period_samp / sample_rate);

    f.ofdm_detected = detectOfdm(x, sample_rate_sps,
                                  f.ofdm_fft_size_est, f.ofdm_cp_ratio);

    {
        au::QuantityD<au::Hertz> hop_rate{au::hertz(0.0)};
        f.fhss_detected = detectFhss(x, sample_rate_sps, hop_rate);
        f.fhss_hop_rate_hz = hop_rate.in(au::hertz);
    }

    double chirp_rate = 0;
    f.chirp_detected  = detectChirp(x, sample_rate_sps, chirp_rate);
    f.chirp_rate_hz_s = chirp_rate;

    return f;
}

} // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
