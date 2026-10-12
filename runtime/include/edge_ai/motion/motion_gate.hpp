#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <edge_ai/core/frame.hpp>
#include <edge_ai/detector/detection.hpp>

struct MotionConfig{
    size_t width = 160;
    size_t height = 120;

    // Grayscale difference required before a pixel counts as changed.
    uint8_t pixel_threshold = 25;

    // 192 pixels = 1% of 160x120.
    size_t minimum_changed_area = 192;

    // Prevents continuous movement from generating an event on every frame.
    uint32_t cooldown_ms = 250;

    // Startup frames that only update the reference (exposure/decoder transients).
    uint32_t warmup_frames = 3;

    // Margin around tracked boxes (fraction of box size) still treated as explained motion.
    float box_margin = 0.25f;
};

struct MotionResult{
    float score = 0.0f;          // changed_pixels / total pixels
    size_t changed_pixels = 0;
    bool active = false;         // changed_pixels >= minimum_changed_area
    bool event = false;          // active and cooldown elapsed
    float processing_ms = 0.0f;
};

// Cheap grayscale frame-difference motion signal.
//
// It answers only "did something meaningfully change?"; it never classifies the change
// and never decides whether inference runs.
class MotionGate{
    public:
        explicit MotionGate(const MotionConfig& config = MotionConfig{});

        MotionResult update(const ImageView& view, uint64_t timestamp_ns);
        void reset();

        // Fraction of changed pixels from the last update that fall outside every box
        // (boxes in original-frame pixels, expanded by box_margin). 0 when nothing changed.
        float changedFractionOutside(const std::vector<Detection>& boxes) const;

        // Changed pixels from the last update inside one box (original-frame pixels, no margin).
        size_t changedPixelsInside(const Detection& box) const;

        // Last binary changed-pixel mask (width x height, 1 = changed).
        const std::vector<uint8_t>& mask() const { return mask_; }

    private:
        void downsampleGray(const ImageView& view);

        MotionConfig config_;

        std::vector<uint8_t> current_;
        std::vector<uint8_t> previous_;
        std::vector<uint8_t> mask_;
        bool has_previous_ = false;

        size_t source_width_ = 0;
        size_t source_height_ = 0;
        size_t last_changed_pixels_ = 0;

        uint32_t frames_seen_ = 0;
        std::optional<uint64_t> last_event_ns_;
        std::optional<uint64_t> last_timestamp_ns_;
};
