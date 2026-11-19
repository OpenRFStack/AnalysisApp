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
 * @file IqReceive.hpp
 * @brief Contiguous IQ capture from a SdrResourceManager UDP stream (ICD §10).
 *
 * The classifier needs one gap-free block: features such as cyclic-prefix
 * correlation, cumulants or the spectral flatness are corrupted by a phase
 * discontinuity spliced into the block. receiveContiguousIq() therefore
 * restarts the block on
 *   - a sequence gap (lost UDP packets),
 *   - an IQ_FLAG_OVERFLOW packet (hardware dropped samples),
 *   - IQ_FLAG_DWELL_CHANGE or a change of centre frequency / sample rate,
 * and reports the sample rate and centre frequency carried by the packets —
 * the ground truth for the samples actually delivered.
 */
#include <sdr/Types.hpp>
#include <sys/select.h>
#include <sys/socket.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

namespace analysis
{

struct IqCapture
{
    std::vector<float> iq;                    ///< interleaved CF32, contiguous
    double             sample_rate_hz = 0.0;  ///< from the packet headers (0 = no packet)
    double             center_freq_hz = 0.0;  ///< from the packet headers
    uint64_t           packets        = 0;
    uint64_t           gaps           = 0;  ///< sequence discontinuities
    uint64_t           lost_packets   = 0;
    uint64_t           overflows      = 0;
    uint64_t           restarts       = 0;  ///< times the block was restarted
};

/// Receive up to @p target_samples contiguous samples within @p timeout_ms.
/// On timeout the longest contiguous block received so far is returned.
inline IqCapture receiveContiguousIq(int udp_fd, int target_samples, int timeout_ms)
{
    using namespace std::chrono;
    IqCapture cap, best;
    cap.iq.reserve((size_t)target_samples * 2);
    auto                 deadline = steady_clock::now() + milliseconds(timeout_ms);
    std::vector<uint8_t> buf(65'536);
    bool                 have_seq = false;
    uint32_t             last_seq = 0;
    auto                 restart  = [&]
    {
        if (cap.iq.size() > best.iq.size())
            best.iq = cap.iq;
        cap.iq.clear();
        ++cap.restarts;
    };
    while ((int)(cap.iq.size() / 2) < target_samples)
    {
        auto now = steady_clock::now();
        if (now >= deadline)
            break;
        long   remaining_ms = duration_cast<milliseconds>(deadline - now).count();
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(udp_fd, &fds);
        timeval tv{remaining_ms / 1000, (remaining_ms % 1000) * 1000};
        if (::select(udp_fd + 1, &fds, nullptr, nullptr, &tv) <= 0)
            break;
        ssize_t n = ::recv(udp_fd, buf.data(), buf.size(), 0);
        if (n < (ssize_t)sdr::IQ_PACKET_HEADER_SIZE)
            continue;
        sdr::IqPacketHeader hdr;
        std::memcpy(&hdr, buf.data(), sizeof hdr);
        if (hdr.magic != sdr::IQ_PACKET_MAGIC)
            continue;
        const size_t n_samp = hdr.num_samples;
        if ((size_t)n < sizeof hdr + n_samp * 8)
            continue;  // truncated
        ++cap.packets;

        bool discontinuity = false;
        if (have_seq && !(hdr.flags & sdr::IQ_FLAG_FIRST_PACKET))
        {
            const int32_t diff = (int32_t)(hdr.sequence - (last_seq + 1));
            if (diff < 0)
                continue;  // late / duplicate
            if (diff > 0)
            {
                ++cap.gaps;
                cap.lost_packets += (uint64_t)diff;
                discontinuity = true;
            }
        }
        have_seq = true;
        last_seq = hdr.sequence;
        if (hdr.flags & sdr::IQ_FLAG_OVERFLOW)
        {
            ++cap.overflows;
            discontinuity = true;
        }
        if (hdr.flags & sdr::IQ_FLAG_DWELL_CHANGE)
            discontinuity = true;
        if (cap.sample_rate_hz > 0.0
            && ((double)hdr.sample_rate != cap.sample_rate_hz
                || (double)hdr.center_freq_hz != cap.center_freq_hz))
            discontinuity = true;
        if (discontinuity && !cap.iq.empty())
            restart();
        cap.sample_rate_hz = (double)hdr.sample_rate;
        cap.center_freq_hz = (double)hdr.center_freq_hz;

        const float* s = reinterpret_cast<const float*>(buf.data() + sizeof hdr);
        cap.iq.insert(cap.iq.end(), s, s + n_samp * 2);
    }
    if ((int)(cap.iq.size() / 2) > target_samples)
        cap.iq.resize((size_t)target_samples * 2);
    if (best.iq.size() > cap.iq.size())
        cap.iq = std::move(best.iq);  // timed out mid-block
    return cap;
}

}  // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
