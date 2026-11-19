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
// Contiguous IQ capture over real loopback UDP, and the detection admission
// policy. No broker, no radio.
#include "DetectionAdmission.hpp"
#include "IqReceive.hpp"

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <complex>
#include <vector>

using namespace analysis;

namespace
{
struct Loop
{
    int rx = -1, tx = -1;

    Loop()
    {
        rx = ::socket(AF_INET, SOCK_DGRAM, 0);
        sockaddr_in a{};
        a.sin_family      = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        ::bind(rx, (sockaddr*)&a, sizeof a);
        socklen_t l = sizeof a;
        ::getsockname(rx, (sockaddr*)&a, &l);
        tx = ::socket(AF_INET, SOCK_DGRAM, 0);
        ::connect(tx, (sockaddr*)&a, sizeof a);
    }

    ~Loop()
    {
        ::close(rx);
        ::close(tx);
    }

    void send(uint32_t seq,
              int      n,
              float    v,
              uint8_t  flags = 0,
              uint32_t sr    = 250'000,
              uint64_t cf    = 433'920'000)
    {
        sdr::IqPacketHeader h{};
        h.magic          = sdr::IQ_PACKET_MAGIC;
        h.sequence       = seq;
        h.sample_rate    = sr;
        h.center_freq_hz = cf;
        h.num_samples    = (uint16_t)n;
        h.flags          = flags;
        std::vector<uint8_t> b(sizeof h + (size_t)n * 8);
        std::memcpy(b.data(), &h, sizeof h);
        auto* s = reinterpret_cast<float*>(b.data() + sizeof h);
        for (int i = 0; i < 2 * n; ++i)
            s[i] = v;
        ::send(tx, b.data(), b.size(), 0);
    }
};
}  // namespace

TEST(IqReceive, ReportsTheRateCarriedByThePackets)
{
    Loop l;
    l.send(0, 100, 1.f, sdr::IQ_FLAG_FIRST_PACKET);
    l.send(1, 100, 1.f);
    auto cap = receiveContiguousIq(l.rx, 200, 500);
    EXPECT_EQ(cap.iq.size(), 400u);
    EXPECT_DOUBLE_EQ(cap.sample_rate_hz, 250'000.0);
    EXPECT_DOUBLE_EQ(cap.center_freq_hz, 433.92e6);
}

// A lost packet must not be spliced into the block handed to the classifier.
TEST(IqReceive, SequenceGapRestartsTheBlock)
{
    Loop l;
    l.send(0, 100, 1.f, sdr::IQ_FLAG_FIRST_PACKET);
    l.send(3, 100, 2.f);  // packets 1–2 lost
    l.send(4, 100, 2.f);
    auto cap = receiveContiguousIq(l.rx, 200, 500);
    ASSERT_EQ(cap.iq.size(), 400u);
    for (float v : cap.iq)
        EXPECT_EQ(v, 2.f) << "pre-gap samples spliced in";
    EXPECT_EQ(cap.gaps, 1u);
    EXPECT_EQ(cap.lost_packets, 2u);
}

TEST(IqReceive, OverflowAndRetuneRestartTheBlock)
{
    Loop l;
    l.send(0, 100, 1.f, sdr::IQ_FLAG_FIRST_PACKET);
    l.send(1, 0, 0.f, sdr::IQ_FLAG_OVERFLOW);  // flag-only overflow packet
    l.send(2, 100, 2.f);
    l.send(3, 100, 3.f, sdr::IQ_FLAG_DWELL_CHANGE, 250'000, 434'000'000);
    l.send(4, 100, 3.f, 0, 250'000, 434'000'000);
    auto cap = receiveContiguousIq(l.rx, 200, 500);
    ASSERT_EQ(cap.iq.size(), 400u);
    for (float v : cap.iq)
        EXPECT_EQ(v, 3.f);
    EXPECT_DOUBLE_EQ(cap.center_freq_hz, 434e6);
    EXPECT_EQ(cap.overflows, 1u);
}

TEST(IqReceive, TimeoutReturnsTheLongestContiguousRun)
{
    Loop l;
    l.send(0, 150, 1.f, sdr::IQ_FLAG_FIRST_PACKET);
    l.send(1, 150, 1.f);  // 300 contiguous
    l.send(5, 50, 2.f);   // gap, then only 50
    auto cap = receiveContiguousIq(l.rx, 1000, 200);
    EXPECT_EQ(cap.iq.size(), 600u);
    for (float v : cap.iq)
        EXPECT_EQ(v, 1.f);
}

// ANA-02: a detection dropped because the queue was full must not start the
// 5-minute cooldown.
TEST(DetectionAdmission, FullQueueDoesNotStartTheCooldown)
{
    DetectionAdmission adm(300'000, 6);
    EXPECT_FALSE(adm.admit(433.92e6, 1000, /*queue_size=*/6));  // full → dropped
    EXPECT_TRUE(adm.admit(433.92e6, 2000, /*queue_size=*/0))
        << "frequency locked out although it was never analysed";
}

TEST(DetectionAdmission, CooldownSuppressesRepeatsThenExpires)
{
    DetectionAdmission adm(300'000, 6);
    EXPECT_TRUE(adm.admit(433.92e6, 0, 0));
    EXPECT_FALSE(adm.admit(433.93e6, 1000, 0));    // same 100 kHz bucket
    EXPECT_TRUE(adm.admit(434.92e6, 1000, 0));     // other bucket
    EXPECT_TRUE(adm.admit(433.92e6, 300'001, 0));  // cooldown over
}

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
