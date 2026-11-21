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
// IqTaskChannel against an in-process loopback AMQP peer that can be
// restarted on the same port (no broker involved).

#include "IqTaskChannel.hpp"

#include <gtest/gtest.h>
#include <proton/listen_handler.hpp>
#include <proton/listener.hpp>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace
{

/// Loopback AMQP peer: accepts the channel's request sender and dynamic
/// reply receiver, records message bodies and link counts.
class TestPeer
    : public proton::messaging_handler
    , public proton::listen_handler
{
public:
    explicit TestPeer(int port = 0)
        : want_port_(port)
    {
        thread_ = std::thread(
            [this]
            {
                container_.run();
            });
        std::unique_lock lk(mu_);
        cv_.wait_for(lk,
                     5s,
                     [this]
                     {
                         return port_ != 0 || failed_;
                     });
    }

    ~TestPeer()
    {
        container_.stop();
        if (thread_.joinable())
            thread_.join();
    }

    int port() const
    {
        return port_;
    }

    int requestLinks()
    {
        std::lock_guard lk(mu_);
        return request_links_;
    }

    int replyLinks()
    {
        std::lock_guard lk(mu_);
        return reply_links_;
    }

    size_t waitFor(size_t n, std::chrono::milliseconds timeout)
    {
        std::unique_lock lk(mu_);
        cv_.wait_for(lk,
                     timeout,
                     [&]
                     {
                         return bodies_.size() >= n;
                     });
        return bodies_.size();
    }

    std::vector<std::string> bodies()
    {
        std::lock_guard lk(mu_);
        return bodies_;
    }

    void on_open(proton::listener& l) override
    {
        std::lock_guard lk(mu_);
        port_ = l.port();
        cv_.notify_all();
    }

    void on_error(proton::listener&, const std::string&) override
    {
        std::lock_guard lk(mu_);
        failed_ = true;
        cv_.notify_all();
    }

    void on_container_start(proton::container& c) override
    {
        listener_ = c.listen("127.0.0.1:" + std::to_string(want_port_), *this);
    }

    void on_receiver_open(proton::receiver& r) override
    {
        r.open();
        std::lock_guard lk(mu_);
        ++request_links_;
    }

    void on_sender_open(proton::sender& s) override
    {
        s.open();
        std::lock_guard lk(mu_);
        ++reply_links_;
    }

    void on_error(const proton::error_condition&) override
    {
    }

    void on_message(proton::delivery&, proton::message& m) override
    {
        std::lock_guard lk(mu_);
        bodies_.push_back(proton::get<std::string>(m.body()));
        cv_.notify_all();
    }

private:
    int                      want_port_;
    std::mutex               mu_;
    std::condition_variable  cv_;
    int                      port_{0};
    bool                     failed_{false};
    int                      request_links_{0};
    int                      reply_links_{0};
    std::vector<std::string> bodies_;
    proton::listener         listener_;
    proton::container        container_{*this};
    std::thread              thread_;
};

std::string url(int port)
{
    return "amqp://127.0.0.1:" + std::to_string(port);
}

bool waitUntil(const std::function<bool()>& pred, std::chrono::milliseconds timeout)
{
    const auto end = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < end)
    {
        if (pred())
            return true;
        std::this_thread::sleep_for(5ms);
    }
    return pred();
}

}  // namespace

// TASK_STOP sent while the channel reconnects used to be dropped silently; the
// RM then held the capture task (and its radio) for the full task duration.
TEST(IqTaskChannel, TaskStopDuringReconnectIsDelivered)
{
    auto      peer = std::make_unique<TestPeer>();
    const int port = peer->port();
    ASSERT_NE(port, 0);
    analysis::IqTaskChannel ch(url(port), "", "", "sdr.tasks");
    ch.start(3000);
    ch.send(R"({"msg_type":"TASK_STOP","task_id":"a"})");
    ASSERT_EQ(peer->waitFor(1, 2s), 1u);

    peer.reset();  // broker restarts
    std::this_thread::sleep_for(300ms);
    ch.send(R"({"msg_type":"TASK_STOP","task_id":"b"})");

    peer = std::make_unique<TestPeer>(port);
    ASSERT_EQ(peer->port(), port);
    const size_t got = peer->waitFor(1, 8s);
    std::printf("TASK_STOP during reconnect: %zu of 1 delivered\n", got);
    ASSERT_EQ(got, 1u);
    EXPECT_NE(peer->bodies()[0].find("\"b\""), std::string::npos);
}

// proton restores the links on reconnect; on_connection_open opened another
// request sender and reply receiver next to them after every broker restart.
TEST(IqTaskChannel, OneLinkEachAfterBrokerRestart)
{
    auto      peer = std::make_unique<TestPeer>();
    const int port = peer->port();
    ASSERT_NE(port, 0);
    analysis::IqTaskChannel ch(url(port), "", "", "sdr.tasks");
    ch.start(3000);
    ASSERT_EQ(peer->requestLinks(), 1);

    peer.reset();
    std::this_thread::sleep_for(300ms);
    peer = std::make_unique<TestPeer>(port);
    ASSERT_TRUE(waitUntil(
        [&]
        {
            return peer->requestLinks() >= 1 && peer->replyLinks() >= 1;
        },
        8s));
    std::this_thread::sleep_for(300ms);  // a duplicate opens in the same batch
    std::printf("after restart: %d request sender(s), %d reply receiver(s)\n",
                peer->requestLinks(),
                peer->replyLinks());
    EXPECT_EQ(peer->requestLinks(), 1);
    EXPECT_EQ(peer->replyLinks(), 1);
}

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
