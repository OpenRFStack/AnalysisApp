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
#include "IqCollector.hpp"

#include "IqReceive.hpp"
#include "IqTaskChannel.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <proton/connection.hpp>
#include <proton/connection_options.hpp>
#include <proton/container.hpp>
#include <proton/delivery.hpp>
#include <proton/message.hpp>
#include <proton/messaging_handler.hpp>
#include <proton/receiver.hpp>
#include <proton/receiver_options.hpp>
#include <proton/reconnect_options.hpp>
#include <proton/sender.hpp>
#include <proton/sender_options.hpp>
#include <proton/source_options.hpp>
#include <proton/symbol.hpp>
#include <proton/target_options.hpp>
#include <proton/transport.hpp>
#include <proton/work_queue.hpp>
#include <sdr/Types.hpp>
#include <spdlog/spdlog.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <au/prefix.hh>
#include <au/units/seconds.hh>

namespace analysis
{

using json = nlohmann::json;
using namespace std::chrono;

// ── JSON helpers ──────────────────────────────────────────────────────────────

static std::string buildTaskRequestJson(const std::string& request_id,
                                        double             center_freq_hz,
                                        double             bandwidth_hz,
                                        double             sample_rate_sps,
                                        int64_t            duration_ms,
                                        const std::string& dest_ip,
                                        int                rank,
                                        uint16_t           prebound_port = 0)
{
    auto ts        = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    json streaming = {{"dest_ip", dest_ip}};
    if (prebound_port > 0)
        streaming["dest_ports"] = json::array({static_cast<int>(prebound_port)});

    return json{{"msg_type", "TASK_REQUEST"},
                {"schema_version", sdr::SCHEMA_VERSION},
                {"request_id", request_id},
                {"timestamp_ms", ts},
                {"task_type", "NARROWBAND"},
                {"rank", rank},
                {"schedule", {{"mode", "IMMEDIATE"}, {"duration_ms", duration_ms}}},
                {"rf",
                 {{"center_freq_hz", center_freq_hz},
                  {"bandwidth_hz", bandwidth_hz},
                  {"sample_rate_sps", sample_rate_sps},
                  {"rx_count", 1}}},
                {"streaming", streaming}}
        .dump();
}

static std::string buildTaskStopJson(const std::string& request_id, const std::string& task_id)
{
    return json{{"msg_type", "TASK_STOP"},
                {"request_id", request_id},
                {"task_id", task_id},
                {"timestamp_ms",
                 duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()},
                {"reason", "collection complete"}}
        .dump();
}

// ── UDP helpers ───────────────────────────────────────────────────────────────

static int openBoundUdpSocket(int port)
{
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0)
        return -1;
    int rcvbuf = 16 * 1024 * 1024;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = INADDR_ANY;
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        ::close(fd);
        return -1;
    }
    return fd;
}

IqCollector::IqCollector(const AmqpConfig&      amqp_cfg,
                         const CollectorConfig& col_cfg,
                         const std::string&     local_ip,
                         int                    task_rank)
    : amqp_cfg_(amqp_cfg)
    , col_cfg_(col_cfg)
    , local_ip_(local_ip)
    , task_rank_(task_rank)
{
    ch_ = std::make_unique<IqTaskChannel>(
        amqp_cfg_.url, amqp_cfg_.username, amqp_cfg_.password, amqp_cfg_.task_request_queue);
    ch_->start(60'000);
}

IqCollector::~IqCollector() = default;

std::vector<float> IqCollector::collect(au::QuantityD<au::Hertz> center_freq_hz,
                                        au::QuantityD<au::Hertz> bandwidth_hz,
                                        const std::string&       request_id)
{
    // Serialize: only one NARROWBAND task at a time (SDR is an exclusive resource).
    std::lock_guard<std::mutex> collect_guard(collect_mu_);

    const double bw = std::max(bandwidth_hz.in(au::hertz), 25000.0);
    // ANA-07: a capture sized to the detection (was always 2 MS/s — a 12.5 kHz
    // signal filled 0.6 % of it and every feature was the out-of-band noise).
    const double sr = collectSampleRate(col_cfg_, bandwidth_hz.in(au::hertz));
    int64_t dur_ms  = static_cast<int64_t>((double)col_cfg_.collect_samples / sr * 1000.0) + 500;

    // Pre-bind UDP socket before submitting the task — controller streams
    // to the ready socket immediately without the 30s AMQP-latency data-loss race.
    int      prebound_fd   = openBoundUdpSocket(0);
    uint16_t prebound_port = 0;
    if (prebound_fd >= 0)
    {
        struct sockaddr_in sa{};
        socklen_t          sl = sizeof(sa);
        if (getsockname(prebound_fd, reinterpret_cast<sockaddr*>(&sa), &sl) == 0)
            prebound_port = ntohs(sa.sin_port);
    }

    auto closePreboundFd = [&]
    {
        if (prebound_fd >= 0)
        {
            ::close(prebound_fd);
            prebound_fd = -1;
        }
    };

    // Step 1: submit NARROWBAND task via persistent channel.
    std::string req_json = buildTaskRequestJson(request_id,
                                                center_freq_hz.in(au::hertz),
                                                bw,
                                                sr,
                                                dur_ms,
                                                local_ip_,
                                                task_rank_,
                                                prebound_port);

    spdlog::info("IqCollector: submitting NARROWBAND {:.3f} MHz req={}",
                 center_freq_hz.in(au::mega(au::hertz)),
                 request_id);

    const int timeout_ms =
        static_cast<int>(col_cfg_.analysis_timeout_ms.in(au::milli(au::seconds)));
    std::string resp_body = ch_->exchange(req_json, request_id, timeout_ms);
    if (resp_body.empty())
    {
        spdlog::error("IqCollector: no response for req={}", request_id);
        closePreboundFd();
        return {};
    }

    // Step 2: parse TASK_ACCEPTED.
    std::string accepted_task_id;
    int         udp_port = 0;
    try
    {
        auto j = json::parse(resp_body);
        if (j.value("status", "") != "ACCEPTED")
        {
            spdlog::warn("IqCollector: task rejected: {}", j.value("reject_reason", "unknown"));
            closePreboundFd();
            return {};
        }
        accepted_task_id = j.value("task_id", "");
        if (j.contains("streams") && !j["streams"].empty())
        {
            udp_port = j["streams"][0].value("udp_port", 0);
            last_sr_ = au::hertz(j["streams"][0].value("sample_rate_sps", sr));
        }
        else
        {
            last_sr_ = au::hertz(sr);
        }
        if (udp_port == 0)
        {
            spdlog::error("IqCollector: ACCEPTED missing streams[0].udp_port");
            closePreboundFd();
            return {};
        }
        spdlog::debug("IqCollector: accepted task_id={} port={} sr={:.0f}",
                      accepted_task_id,
                      udp_port,
                      last_sr_.in(au::hertz));
    }
    catch (const std::exception& ex)
    {
        spdlog::error("IqCollector: parse error: {}", ex.what());
        closePreboundFd();
        return {};
    }

    // Step 3: use pre-bound socket if controller honoured our port, else re-bind.
    int udp_fd;
    if (prebound_fd >= 0 && prebound_port > 0 && udp_port == prebound_port)
    {
        udp_fd      = prebound_fd;
        prebound_fd = -1;
    }
    else
    {
        closePreboundFd();
        udp_fd = openBoundUdpSocket(udp_port);
        if (udp_fd < 0)
        {
            spdlog::error("IqCollector: bind port {} failed: {}", udp_port, strerror(errno));
            return {};
        }
    }

    // Step 4: receive IQ.
    // One gap-free block; the packet headers carry the rate of the samples
    // actually delivered (the response used to report the device's wideband
    // rate for DDC-decimated streams, skewing every rate-derived feature).
    auto cap = receiveContiguousIq(udp_fd, col_cfg_.collect_samples, timeout_ms);
    ::close(udp_fd);
    if (cap.sample_rate_hz > 0.0)
    {
        if (std::abs(cap.sample_rate_hz - last_sr_.in(au::hertz)) > 1.0)
            spdlog::warn(
                "IqCollector: stream delivers {:.0f} S/s but the response said {:.0f} S/s — "
                "using the packet headers",
                cap.sample_rate_hz,
                last_sr_.in(au::hertz));
        last_sr_ = au::hertz(cap.sample_rate_hz);
    }
    if (cap.gaps || cap.overflows)
        spdlog::info(
            "IqCollector: {} gap(s) ({} pkts lost), {} overflow(s) — {} restart(s) to keep "
            "the block contiguous",
            cap.gaps,
            cap.lost_packets,
            cap.overflows,
            cap.restarts);
    std::vector<float> iq = std::move(cap.iq);
    spdlog::info("IqCollector: collected {} samples (sr={:.0f} Hz)",
                 (int)iq.size() / 2,
                 last_sr_.in(au::hertz));

    // Step 5: fire-and-forget TASK_STOP.
    if (!accepted_task_id.empty())
        ch_->send(buildTaskStopJson(request_id + "_stop", accepted_task_id));

    return iq;
}

}  // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
