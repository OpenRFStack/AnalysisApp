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
 * @file DetectionAdmission.hpp
 * @brief Re-analysis cooldown + queue-depth policy for incoming detections.
 *
 * Extracted from AnalysisService::subscriptionLoop() so it can be unit
 * tested without a broker. Not thread-safe: the caller holds its queue mutex.
 *
 * The cooldown is recorded only when a detection is actually enqueued. It
 * used to be recorded before the queue-depth check, so a detection dropped
 * because the queue was full still locked its frequency out for the whole
 * cooldown (5 min) — contrary to the intent that "the signal will be
 * re-detected and enqueued on the next sweep pass".
 */
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <unordered_map>

namespace analysis
{

class DetectionAdmission
{
public:
    DetectionAdmission(int64_t cooldown_ms, size_t max_queue, double bucket_hz = 100'000.0)
        : cooldown_ms_(cooldown_ms)
        , max_queue_(max_queue)
        , bucket_hz_(bucket_hz)
    {
    }

    /// True → enqueue it (the cooldown starts now). False → drop it.
    bool admit(double center_freq_hz, int64_t now_ms, size_t queue_size)
    {
        const int64_t bucket = (int64_t)(center_freq_hz / bucket_hz_);
        auto          it     = last_.find(bucket);
        if (it != last_.end() && now_ms - it->second < cooldown_ms_)
            return false;
        if (queue_size >= max_queue_)
            return false;  // no cooldown: retry next pass
        last_[bucket] = now_ms;
        if (last_.size() > 4096)
            prune(now_ms);
        return true;
    }

private:
    int64_t                              cooldown_ms_;
    size_t                               max_queue_;
    double                               bucket_hz_;
    std::unordered_map<int64_t, int64_t> last_;

    void prune(int64_t now_ms)
    {  // bound memory on wide sweeps
        for (auto it = last_.begin(); it != last_.end();)
            it = (now_ms - it->second >= cooldown_ms_) ? last_.erase(it) : std::next(it);
    }
};

}  // namespace analysis

/*
========================================================================
End of file — OpenRFStack
Subject to Personal Use License
https://github.com/OpenRFStack
========================================================================
*/
