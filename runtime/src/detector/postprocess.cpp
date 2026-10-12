#include <edge_ai/detector/postprocess.hpp>

#include <algorithm>
#include <stdexcept>

float boxIou(const Detection& a, const Detection& b){
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);

    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw*ih;

    const float area_a = std::max(0.0f, a.x2 - a.x1)*std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1)*std::max(0.0f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;

    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<Detection> nonMaxSuppression(std::vector<Detection> boxes, float iou_threshold, size_t max_detections){
    std::stable_sort(boxes.begin(), boxes.end(), [](const Detection& a, const Detection& b){
        return a.score > b.score;
    });

    std::vector<Detection> kept;
    std::vector<bool> suppressed(boxes.size(), false);

    for(size_t i=0; i<boxes.size() && kept.size()<max_detections; i++){
        if(suppressed[i]) continue;
        kept.push_back(boxes[i]);

        for(size_t j=i+1; j<boxes.size(); j++){
            if(!suppressed[j] && boxIou(boxes[i], boxes[j]) > iou_threshold){
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

std::vector<Detection> postprocessYolov8(
    const float* output,
    size_t num_attrs,
    size_t num_candidates,
    const PostprocessConfig& config,
    const PreprocessMetadata& meta
){
    if(output == nullptr) throw std::invalid_argument("null detector output");
    if(num_attrs < 5 || config.class_id >= num_attrs - 4){
        throw std::invalid_argument("class id outside detector output");
    }

    const float* cx = output;
    const float* cy = output + num_candidates;
    const float* w = output + 2*num_candidates;
    const float* h = output + 3*num_candidates;
    const float* score = output + (4 + config.class_id)*num_candidates;

    // Boxes stay in model space for NMS; this matches Ultralytics ordering.
    std::vector<Detection> candidates;
    for(size_t i=0; i<num_candidates; i++){
        if(score[i] < config.confidence) continue;

        const float hw = w[i]*0.5f;
        const float hh = h[i]*0.5f;
        candidates.push_back({cx[i]-hw, cy[i]-hh, cx[i]+hw, cy[i]+hh, score[i]});
    }

    std::vector<Detection> kept = nonMaxSuppression(std::move(candidates), config.iou, config.max_detections);

    for(Detection& d : kept){
        modelToOriginal(d.x1, d.y1, meta);
        modelToOriginal(d.x2, d.y2, meta);
    }
    return kept;
}
