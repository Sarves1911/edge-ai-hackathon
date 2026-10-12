#include <edge_ai/tracker/tracker.hpp>

#include <algorithm>
#include <cmath>
#include <tuple>

#include <edge_ai/detector/postprocess.hpp>

namespace {

constexpr double kNsPerSecond = 1e9;

float secondsBetween(uint64_t from_ns, uint64_t to_ns){
    if(to_ns <= from_ns) return 0.0f;
    return static_cast<float>(static_cast<double>(to_ns - from_ns) / kNsPerSecond);
}

} // namespace

Detection Track::box() const{
    return {cx - w*0.5f, cy - h*0.5f, cx + w*0.5f, cy + h*0.5f, score};
}

Tracker::Tracker(const TrackerConfig& config)
    : config_(config)
{
}

void Tracker::reset(){
    tracks_.clear();
    history_.clear();
    current_ts_ns_ = 0;
    have_ts_ = false;
    failed_ = false;
    abrupt_ = false;
}

void Tracker::setFrameSize(size_t width, size_t height){
    frame_width_ = width;
    frame_height_ = height;
}

void Tracker::advance(std::vector<Track>& tracks, uint64_t from_ns, uint64_t to_ns) const{
    const float dt = secondsBetween(from_ns, to_ns);
    if(dt <= 0.0f) return;

    for(Track& t : tracks){
        t.cx += t.vx*dt;
        t.cy += t.vy*dt;
    }
}

void Tracker::correct(std::vector<Track>& tracks, const DetectionBatch& batch, uint64_t at_ns){
    const std::vector<Detection>& dets = batch.detections;

    // Greedy association, best pairs first: IoU, with a center-distance fallback.
    std::vector<std::tuple<float, size_t, size_t>> pairs;
    for(size_t i=0; i<tracks.size(); i++){
        const Detection tb = tracks[i].box();
        for(size_t j=0; j<dets.size(); j++){
            const Detection& d = dets[j];
            const float iou = boxIou(tb, d);

            const float dx = (d.x1 + d.x2)*0.5f - tracks[i].cx;
            const float dy = (d.y1 + d.y2)*0.5f - tracks[i].cy;
            const float scale = std::max(std::max(tracks[i].h, d.y2 - d.y1), 1.0f);
            const float distance = std::hypot(dx, dy) / scale;

            if(iou >= config_.match_iou){
                pairs.emplace_back(1.0f + iou, i, j);
            }
            else if(distance <= config_.match_distance){
                pairs.emplace_back(1.0f - distance/config_.match_distance, i, j);
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b){ return std::get<0>(a) > std::get<0>(b); });

    std::vector<bool> track_matched(tracks.size(), false);
    std::vector<bool> det_matched(dets.size(), false);

    for(const auto& [iou, ti, dj] : pairs){
        if(track_matched[ti] || det_matched[dj]) continue;
        track_matched[ti] = true;
        det_matched[dj] = true;

        Track& t = tracks[ti];
        const Detection& d = dets[dj];

        const float mcx = (d.x1 + d.x2)*0.5f;
        const float mcy = (d.y1 + d.y2)*0.5f;
        const float mw = d.x2 - d.x1;
        const float mh = d.y2 - d.y1;

        const float rx = mcx - t.cx;
        const float ry = mcy - t.cy;
        const float dt = std::max(secondsBetween(t.last_detection_ts_ns, at_ns), 1e-3f);

        const float old_vx = t.vx;
        const float old_vy = t.vy;

        if(t.hits == 1){
            // Two-point initiation: the first velocity estimate comes straight from the
            // displacement between the first two observations.
            t.vx += rx/dt;
            t.vy += ry/dt;
            t.cx = mcx;
            t.cy = mcy;
        }
        else{
            t.cx += config_.position_gain*rx;
            t.cy += config_.position_gain*ry;
            t.vx += config_.velocity_gain*rx/dt;
            t.vy += config_.velocity_gain*ry/dt;
        }
        t.w += config_.size_gain*(mw - t.w);
        t.h += config_.size_gain*(mh - t.h);

        // Only judge abruptness once a velocity exists (not on two-point initiation).
        if(t.confirmed && t.hits >= 2){
            const float dv = std::hypot(t.vx - old_vx, t.vy - old_vy);
            if(dv / std::max(t.h, 1.0f) > config_.abrupt_velocity_change) abrupt_ = true;
        }

        t.hits++;
        t.misses = 0;
        t.score = d.score;
        t.last_detection_ts_ns = at_ns;
        if(t.hits >= static_cast<int>(config_.min_hits) || d.score >= config_.confirm_score) t.confirmed = true;
    }

    for(size_t i=0; i<tracks.size(); i++){
        if(!track_matched[i]) tracks[i].misses++;
    }

    for(size_t j=0; j<dets.size(); j++){
        if(det_matched[j]) continue;
        const Detection& d = dets[j];

        Track t;
        t.id = next_id_++;
        t.cx = (d.x1 + d.x2)*0.5f;
        t.cy = (d.y1 + d.y2)*0.5f;
        t.w = d.x2 - d.x1;
        t.h = d.y2 - d.y1;
        t.hits = 1;
        t.score = d.score;
        t.last_detection_ts_ns = at_ns;
        t.confirmed = config_.min_hits <= 1 || d.score >= config_.confirm_score;
        tracks.push_back(t);
    }
}

void Tracker::prune(std::vector<Track>& tracks, uint64_t now_ns){
    const uint64_t max_age_ns = static_cast<uint64_t>(config_.max_track_age_ms)*1'000'000ULL;

    auto remove = [&](const Track& t){
        // Tentative tracks die on their first miss; confirmed ones get max_misses.
        bool stale = t.confirmed
            ? t.misses >= static_cast<int>(config_.max_misses)
            : t.misses >= 1;

        if(now_ns > t.last_detection_ts_ns && now_ns - t.last_detection_ts_ns > max_age_ns) stale = true;

        if(frame_width_ > 0 && frame_height_ > 0){
            const Detection b = t.box();
            if(b.x2 <= 0.0f || b.y2 <= 0.0f || b.x1 >= frame_width_ || b.y1 >= frame_height_) stale = true;
        }

        if(stale && t.confirmed) failed_ = true;
        return stale;
    };

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), remove), tracks.end());
}

void Tracker::record(){
    if(!history_.empty() && history_.back().ts_ns == current_ts_ns_){
        history_.back().tracks = tracks_;
    }
    else{
        history_.push_back({current_ts_ns_, tracks_});
    }
    trimHistory();
}

void Tracker::trimHistory(){
    const uint64_t window_ns = static_cast<uint64_t>(config_.history_ms)*1'000'000ULL;

    // Keep the newest snapshot at or before (now - window) so any T inside the window
    // still has a state at or before it.
    while(history_.size() > 1 && history_[1].ts_ns + window_ns <= current_ts_ns_){
        history_.pop_front();
    }
}

void Tracker::predict(uint64_t ts_ns, const StillTest& is_still){
    if(!have_ts_ || ts_ns < current_ts_ns_){
        if(have_ts_) reset();
        current_ts_ns_ = ts_ns;
        have_ts_ = true;
        record();
        return;
    }

    if(is_still){
        // Nothing changed where the person is: hold position and bleed off velocity so a
        // stopped person is not carried forward by constant-velocity prediction.
        const float dt = secondsBetween(current_ts_ns_, ts_ns);
        for(Track& t : tracks_){
            if(is_still(t)){
                t.vx *= 0.5f;
                t.vy *= 0.5f;
            }
            else{
                t.cx += t.vx*dt;
                t.cy += t.vy*dt;
            }
        }
    }
    else{
        advance(tracks_, current_ts_ns_, ts_ns);
    }
    current_ts_ns_ = ts_ns;
    prune(tracks_, ts_ns);
    record();
}

void Tracker::update(const DetectionBatch& batch){
    const uint64_t t_obs = batch.frame_ts_ns;
    abrupt_ = false;

    if(!have_ts_){
        current_ts_ns_ = t_obs;
        have_ts_ = true;
    }

    // Observation for the present (synchronous detector) or newer frame.
    if(t_obs >= current_ts_ns_){
        advance(tracks_, current_ts_ns_, t_obs);
        current_ts_ns_ = t_obs;
        correct(tracks_, batch, t_obs);
        prune(tracks_, t_obs);
        record();
        return;
    }

    // Delayed observation: find the historical state at or before the source frame.
    auto it = std::find_if(history_.rbegin(), history_.rend(), [&](const Snapshot& s){
        return s.ts_ns <= t_obs;
    });

    if(it == history_.rend()){
        // Older than the history window: fall back to latest-result correction.
        ++stale_batches_;
        correct(tracks_, batch, current_ts_ns_);
        prune(tracks_, current_ts_ns_);
        record();
        return;
    }

    // Snapshot timestamps after the observation, re-propagated below.
    std::vector<uint64_t> later;
    for(auto fwd = it.base(); fwd != history_.end(); ++fwd) later.push_back(fwd->ts_ns);

    std::vector<Track> state = it->tracks;
    advance(state, it->ts_ns, t_obs);
    correct(state, batch, t_obs);
    prune(state, t_obs);

    history_.erase(it.base(), history_.end());
    if(history_.back().ts_ns == t_obs) history_.back().tracks = state;
    else history_.push_back({t_obs, state});

    // Re-predict forward through every later frame up to the present.
    uint64_t prev = t_obs;
    for(uint64_t ts : later){
        advance(state, prev, ts);
        prune(state, ts);
        if(history_.back().ts_ns == ts) history_.back().tracks = state;
        else history_.push_back({ts, state});
        prev = ts;
    }

    tracks_ = std::move(state);
    trimHistory();
}

std::vector<Detection> Tracker::boxes(bool confirmed_only) const{
    std::vector<Detection> out;
    for(const Track& t : tracks_){
        if(!confirmed_only || t.confirmed) out.push_back(t.box());
    }
    return out;
}

size_t Tracker::confirmedCount() const{
    return static_cast<size_t>(std::count_if(tracks_.begin(), tracks_.end(), [](const Track& t){ return t.confirmed; }));
}

bool Tracker::weak() const{
    return std::any_of(tracks_.begin(), tracks_.end(), [this](const Track& t){
        return (t.confirmed && t.misses > 0) || t.hits < static_cast<int>(config_.young_hits);
    });
}
