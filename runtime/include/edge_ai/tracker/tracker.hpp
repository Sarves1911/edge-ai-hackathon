#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <vector>

#include <edge_ai/detector/detection.hpp>

struct TrackerConfig{
    float match_iou = 0.3f;

    // Fallback association when boxes barely overlap (fast movers, sparse detector calls):
    // center distance as a fraction of box height.
    float match_distance = 0.5f;

    // Hits required before a track is confirmed, unless one detection is this confident.
    uint32_t min_hits = 2;
    float confirm_score = 0.6f;

    // Detector results in which a track went unmatched before deletion.
    uint32_t max_misses = 3;

    // Track deleted if no detection supported it for this long.
    uint32_t max_track_age_ms = 2000;

    // Alpha-beta filter gains.
    float position_gain = 0.7f;
    float velocity_gain = 0.3f;
    float size_gain = 0.5f;

    // Tracks with fewer hits than this have no reliable velocity yet and need a quick recheck.
    uint32_t young_hits = 3;

    // Changed motion-mask pixels inside a track box at or below which the person is held still.
    uint32_t still_motion_pixels = 2;

    // Velocity change (box heights per second) that counts as abrupt motion.
    float abrupt_velocity_change = 1.0f;

    // Amount of timestamped state history kept for delayed detector results.
    uint32_t history_ms = 1000;
};

struct Track{
    int id = 0;
    float cx = 0.0f;
    float cy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float vx = 0.0f;     // pixels per second
    float vy = 0.0f;
    int hits = 0;
    int misses = 0;
    bool confirmed = false;
    uint64_t last_detection_ts_ns = 0;
    float score = 0.0f;

    Detection box() const;
};

// Constant-velocity person tracker with timestamp-aware correction.
//
// predict() advances tracks to a frame timestamp. update() takes a DetectionBatch whose
// frame_ts_ns may be in the past: the correction is applied to the historical state at
// that timestamp and then re-propagated to the present instead of treating a stale
// observation as current.
class Tracker{
    public:
        explicit Tracker(const TrackerConfig& config = TrackerConfig{});

        void setFrameSize(size_t width, size_t height);

        // Returns true when the scene shows the tracked person did not move this frame.
        using StillTest = std::function<bool(const Track&)>;

        void predict(uint64_t ts_ns, const StillTest& is_still = {});
        void update(const DetectionBatch& batch);

        const std::vector<Track>& tracks() const { return tracks_; }
        std::vector<Detection> boxes(bool confirmed_only = true) const;
        size_t confirmedCount() const;

        uint64_t currentTs() const { return current_ts_ns_; }

        // True once after a confirmed track was lost; cleared by acknowledgeFailure().
        bool failed() const { return failed_; }
        void acknowledgeFailure() { failed_ = false; }

        // A confirmed track went unmatched in the latest detector result, or a track is
        // too young to have a reliable velocity.
        bool weak() const;

        // Latest correction changed a confirmed track's velocity abruptly.
        bool abruptMotion() const { return abrupt_; }

        size_t staleBatches() const { return stale_batches_; }
        size_t historySize() const { return history_.size(); }

        void reset();

    private:
        struct Snapshot{
            uint64_t ts_ns;
            std::vector<Track> tracks;
        };

        void advance(std::vector<Track>& tracks, uint64_t from_ns, uint64_t to_ns) const;
        void correct(std::vector<Track>& tracks, const DetectionBatch& batch, uint64_t at_ns);
        void prune(std::vector<Track>& tracks, uint64_t now_ns);
        void record();
        void trimHistory();

        TrackerConfig config_;
        std::vector<Track> tracks_;
        std::deque<Snapshot> history_;

        uint64_t current_ts_ns_ = 0;
        bool have_ts_ = false;

        size_t frame_width_ = 0;
        size_t frame_height_ = 0;

        int next_id_ = 1;
        bool failed_ = false;
        bool abrupt_ = false;
        size_t stale_batches_ = 0;
};
