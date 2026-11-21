#pragma once
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
#include <spdlog/spdlog.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace analysis
{

// ── Persistent AMQP channel for task request/response ────────────────────────
// Mirrors TaskAmqpChannel in AcquisitionApp: connects once at construction,
// uses a dynamic (unique) receiver so no other subscriber steals responses,
// and reuses the connection across all collect() calls — paying the ~30s
// Artemis settlement cost once instead of on every analysis.

class IqTaskChannel : public proton::messaging_handler
{
public:
    IqTaskChannel(std::string url, std::string user, std::string pass, std::string req_queue)
        : url_(std::move(url))
        , user_(std::move(user))
        , pass_(std::move(pass))
        , req_queue_(std::move(req_queue))
    {
    }

    ~IqTaskChannel()
    {
        stop();
    }

    void start(int connect_timeout_ms = 60'000)
    {
        container_ = std::make_unique<proton::container>(*this);
        thread_    = std::thread(
            [this]
            {
                container_->run();
            });
        std::unique_lock<std::mutex> lk(mu_);
        ready_cv_.wait_for(lk,
                           std::chrono::milliseconds(connect_timeout_ms),
                           [this]
                           {
                               return ready_;
                           });
        if (!ready_)
            spdlog::warn("[IqTaskChannel] not ready after {}ms", connect_timeout_ms);
    }

    void stop()
    {
        if (container_)
        {
            proton::work_queue* wq = nullptr;
            {
                // A TASK_STOP held during a reconnect: give the reconnect
                // (2 s retry delay) a bounded chance to deliver it.
                std::unique_lock<std::mutex> lk(mu_);
                if (!held_.empty()
                    && !ready_cv_.wait_for(lk,
                                           std::chrono::milliseconds(STOP_GRACE_MS),
                                           [this]
                                           {
                                               return held_.empty();
                                           }))
                    spdlog::warn("[IqTaskChannel] shutting down with {} undelivered message(s)",
                                 held_.size());
                wq = wq_;
            }
            if (wq)
                wq->add(
                    [this]
                    {
                        sender_.connection().close();
                    });
            // Always stop the container: if wq_ is stale (transport error,
            // reconnect in progress) the close() above is silently dropped
            // and thread_.join() would hang indefinitely without this.
            container_->stop();
            if (thread_.joinable())
                thread_.join();
            {
                std::lock_guard<std::mutex> lk(mu_);
                wq_ = nullptr;
            }
            container_.reset();
        }
    }

    // Send msg_body and block until the correlated response arrives.
    // Returns {} if timed out. Only one exchange() at a time (enforced by
    // collect_mu_ in IqCollector — do not call concurrently).
    std::string exchange(const std::string& msg_body, const std::string& corr_id, int timeout_ms)
    {
        proton::work_queue* wq = nullptr;
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (!ready_cv_.wait_for(lk,
                                    std::chrono::milliseconds(timeout_ms),
                                    [this]
                                    {
                                        return ready_;
                                    }))
            {
                spdlog::warn("[IqTaskChannel] not ready");
                return {};
            }
            pending_corr_ = corr_id;
            pending_body_.clear();
            pending_done_ = false;
            wq            = wq_;
        }
        wq->add(
            [this, msg_body]() mutable
            {
                proton::message msg;
                msg.body(msg_body);
                msg.content_type("application/json");
                msg.reply_to(reply_addr_);
                // Do NOT check credit() — proton queues when credit arrives.
                if (sender_)
                    sender_.send(msg);
            });
        std::unique_lock<std::mutex> lk(mu_);
        result_cv_.wait_for(lk,
                            std::chrono::milliseconds(timeout_ms),
                            [this]
                            {
                                return pending_done_;
                            });
        return pending_body_;
    }

    // Fire-and-forget (TASK_STOP). While the channel is down (broker restart,
    // reconnect in progress) the message is held and sent when the links are
    // back: a dropped TASK_STOP left the capture task, and its radio, with the
    // RM until the task duration ran out.
    void send(const std::string& msg_body)
    {
        std::unique_lock<std::mutex> lk(mu_);
        if (!ready_ || !wq_)
        {
            if (held_.size() >= MAX_HELD)
                held_.erase(held_.begin());  // keep the newest
            held_.push_back(msg_body);
            spdlog::warn("[IqTaskChannel] not connected; holding message ({} held)", held_.size());
            return;
        }
        wq_->add(
            [this, msg_body]
            {
                sendNow(msg_body);
            });
    }

    // proton callbacks ─────────────────────────────────────────────────────────
    void on_container_start(proton::container& c) override
    {
        proton::connection_options opts;
        if (!user_.empty())
        {
            opts.sasl_allowed_mechs("PLAIN");
            opts.sasl_allow_insecure_mechs(true);
            opts.user(user_).password(pass_);
        }
        else
        {
            opts.sasl_allowed_mechs("ANONYMOUS");
        }
        proton::reconnect_options ropts;
        ropts.delay(proton::duration(2000));
        ropts.max_delay(proton::duration(30'000));
        ropts.max_attempts(0);
        opts.reconnect(ropts);
        c.connect(url_, opts);
    }

    void on_connection_open(proton::connection& conn) override
    {
        // proton restores both links itself on a reconnect (the dynamic
        // receiver gets a fresh address via on_receiver_open); opening them
        // again added a sender and a reply queue per broker restart.
        if (conn.reconnected())
            return;
        proton::sender_options sopts;
        sopts.target(proton::target_options().capabilities({proton::symbol("queue")}));
        sender_ = conn.open_sender(req_queue_, sopts);

        proton::receiver_options ropts;
        ropts.source(proton::source_options().dynamic(true));
        conn.open_receiver("", ropts);
    }

    void on_receiver_open(proton::receiver& r) override
    {
        reply_addr_ = r.source().address();
        spdlog::info("[IqTaskChannel] connected, reply_to={}", reply_addr_);
        std::vector<std::string> held;
        {
            std::lock_guard<std::mutex> lk(mu_);
            wq_    = &r.work_queue();
            ready_ = true;
            held.swap(held_);
            ready_cv_.notify_all();
        }
        for (const auto& body : held)
            sendNow(body);
        if (!held.empty())
            spdlog::info("[IqTaskChannel] sent {} held message(s) after reconnect", held.size());
    }

    void on_message(proton::delivery& d, proton::message& msg) override
    {
        d.accept();
        try
        {
            std::string                 b = proton::get<std::string>(msg.body());
            auto                        j = nlohmann::json::parse(b);
            std::lock_guard<std::mutex> lk(mu_);
            if (!pending_corr_.empty()
                && (j.value("request_id", "") == pending_corr_
                    || j.value("correlation_id", "") == pending_corr_))
            {
                pending_body_ = b;
                pending_done_ = true;
                pending_corr_.clear();
                result_cv_.notify_all();
            }
        }
        catch (...)
        {
        }
    }

    void on_transport_error(proton::transport& t) override
    {
        spdlog::warn("[IqTaskChannel] transport error: {}", t.error().what());
        std::lock_guard<std::mutex> lk(mu_);
        ready_ = false;
        reply_addr_.clear();
    }

    void on_connection_error(proton::connection& c) override
    {
        spdlog::warn("[IqTaskChannel] connection error: {}", c.error().what());
    }

private:
    static constexpr size_t MAX_HELD      = 64;
    static constexpr int    STOP_GRACE_MS = 3000;

    // proton thread only.
    void sendNow(const std::string& body)
    {
        if (!sender_)
            return;
        proton::message msg;
        msg.body(body);
        msg.content_type("application/json");
        sender_.send(msg);
    }

    std::string url_, user_, pass_, req_queue_;

    std::unique_ptr<proton::container> container_;
    std::thread                        thread_;

    proton::sender      sender_;
    proton::work_queue* wq_{nullptr};
    std::string         reply_addr_;

    std::mutex               mu_;
    std::condition_variable  ready_cv_;
    std::condition_variable  result_cv_;
    bool                     ready_{false};
    std::vector<std::string> held_;  ///< sends made while not connected (mu_)
    std::string              pending_corr_;
    std::string              pending_body_;
    bool                     pending_done_{false};
};

}  // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
