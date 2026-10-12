#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <edge_ai/capture/replay_source.hpp>
#include <edge_ai/config/runtime_config.hpp>
#include <edge_ai/core/time.hpp>
#include <edge_ai/detector/simulated_detector.hpp>
#include <edge_ai/metrics/profiler.hpp>
#include <edge_ai/pipeline/pipeline.hpp>

#ifdef EDGE_AI_HAVE_ONNXRUNTIME
#include <edge_ai/detector/onnx_detector.hpp>
#endif

#ifdef EDGE_AI_HAVE_V4L2
#include <edge_ai/capture/v4l2_source.hpp>
#endif

namespace {

std::atomic<bool> g_stop{false};

void onSignal(int){
    g_stop = true;
}

void usage(){
    std::cerr <<
        "usage: edge_ai_runtime [options]\n"
        "  --config <file.json>        runtime configuration (defaults otherwise)\n"
        "  --source <src>              replay:<video> | v4l2:/dev/videoN | <video>\n"
        "  --output <dir>              run directory (frames.csv, detections.jsonl, summary.json)\n"
        "  --mode <m>                  always | fixed_interval | motion_gated | adaptive\n"
        "  --detector <d>              onnx | simulated\n"
        "  --model <file.onnx>         ONNX model path\n"
        "  --threads <n>               ONNX Runtime intra-op threads\n"
        "  --sim-jsonl <file>          cached detections for the simulated detector\n"
        "  --sim-delay-ms <n>          injected detector latency\n"
        "  --sync | --async            run the detector inline or on the worker thread\n"
        "  --busy-policy <p>           skip | latest\n"
        "  --realtime                  pace replay at the recorded frame rate\n"
        "  --max-frames <n>            stop after n frames\n"
        "  --no-overlay                disable the terminal status line\n";
}

std::string requireValue(int& i, int argc, char** argv){
    if(i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

struct CliOptions{
    RuntimeConfig config;
    std::string output_override;
};

CliOptions parseArgs(int argc, char** argv){
    // --config first so explicit flags override file values regardless of order.
    std::string config_path;
    for(int i=1; i<argc; i++){
        if(std::string(argv[i]) == "--config" && i + 1 < argc) config_path = argv[i + 1];
    }

    CliOptions options;
    options.config = config_path.empty() ? RuntimeConfig{} : loadConfig(config_path);
    RuntimeConfig& c = options.config;

    for(int i=1; i<argc; i++){
        const std::string arg = argv[i];
        if(arg == "--config") { ++i; }
        else if(arg == "--source") c.pipeline.source = requireValue(i, argc, argv);
        else if(arg == "--output") c.output.directory = requireValue(i, argc, argv);
        else if(arg == "--mode") c.scheduler.mode = schedulerModeFromString(requireValue(i, argc, argv));
        else if(arg == "--detector") c.pipeline.detector = requireValue(i, argc, argv);
        else if(arg == "--model") c.model.path = requireValue(i, argc, argv);
        else if(arg == "--threads") c.model.threads = std::stoi(requireValue(i, argc, argv));
        else if(arg == "--sim-jsonl") c.pipeline.simulated_jsonl = requireValue(i, argc, argv);
        else if(arg == "--sim-delay-ms") c.pipeline.simulated_delay_ms = static_cast<uint32_t>(std::stoul(requireValue(i, argc, argv)));
        else if(arg == "--sync") c.pipeline.async = false;
        else if(arg == "--async") c.pipeline.async = true;
        else if(arg == "--busy-policy") c.pipeline.busy_policy = busyPolicyFromString(requireValue(i, argc, argv));
        else if(arg == "--realtime") c.pipeline.realtime = true;
        else if(arg == "--max-frames") c.pipeline.max_frames = std::stoull(requireValue(i, argc, argv));
        else if(arg == "--no-overlay") c.output.overlay = false;
        else if(arg == "-h" || arg == "--help"){ usage(); std::exit(0); }
        else throw std::invalid_argument("unknown argument: " + arg);
    }

    validateConfig(c);
    if(c.pipeline.source.empty()) throw std::invalid_argument("--source is required");
    return options;
}

std::unique_ptr<FrameSource> createSource(const RuntimeConfig& c){
    const std::string& source = c.pipeline.source;

    if(source.rfind("v4l2:", 0) == 0){
#ifdef EDGE_AI_HAVE_V4L2
        V4L2Config camera;
        camera.device = source.substr(5);
        camera.width = c.camera.width;
        camera.height = c.camera.height;
        camera.fps = c.camera.fps;
        camera.buffers = c.camera.buffers;
        camera.poll_timeout_ms = c.camera.poll_timeout_ms;
        return std::make_unique<V4L2Source>(camera);
#else
        throw std::invalid_argument("V4L2 capture is only available on Linux");
#endif
    }

    const std::string path = source.rfind("replay:", 0) == 0 ? source.substr(7) : source;
    return std::make_unique<ReplaySource>(path, c.pipeline.realtime ? ReplayMode::REALTIME : ReplayMode::UNLIMITED);
}

std::unique_ptr<Detector> createDetector(const RuntimeConfig& c, Profiler& profiler){
    if(c.pipeline.detector == "simulated"){
        SimulatedDetector::Script script = c.pipeline.simulated_jsonl.empty()
            ? SimulatedDetector::Script([](uint64_t){ return std::vector<Detection>{}; })
            : SimulatedDetector::timelineScript(SimulatedDetector::loadJsonl(c.pipeline.simulated_jsonl));

        profiler.setModelInfo("simulated", 0.0, 0.0f);
        return std::make_unique<SimulatedDetector>(
            c.model.input_width, c.model.input_height, std::move(script), c.pipeline.simulated_delay_ms
        );
    }

#ifdef EDGE_AI_HAVE_ONNXRUNTIME
    PostprocessConfig post;
    post.confidence = c.model.confidence;
    post.iou = c.model.iou;
    post.class_id = c.model.class_id;

    auto detector = std::make_unique<OnnxDetector>(c.model.path, post, c.model.threads);

    const double size_mb = static_cast<double>(std::filesystem::file_size(c.model.path)) / (1024.0*1024.0);
    profiler.setModelInfo(c.model.path, size_mb, detector->modelLoadMs());

    if(c.pipeline.simulated_delay_ms > 0){
        std::cerr << "note: --sim-delay-ms only applies to the simulated detector\n";
    }
    return detector;
#else
    throw std::invalid_argument("built without ONNX Runtime; use --detector simulated");
#endif
}

void printStatus(const PipelineStatus& s, double fps){
    const size_t avoided = s.frames > s.detector_calls ? s.frames - s.detector_calls : 0;
    std::fprintf(
        stderr,
        "STATE %-7s | PERSONS %zu | ACTION %-12s | DETECTOR CALLS %zu | AVOIDED %zu | "
        "LAST INFERENCE AGE %.0f ms | INFERENCE %.1f ms | FPS %.1f\n",
        toString(s.state), s.persons, toString(s.action), s.detector_calls, avoided,
        s.last_result_age_ms < 0.0f ? 0.0f : s.last_result_age_ms, s.last_infer_ms, fps
    );
}

} // namespace

int main(int argc, char** argv){
    try{
        CliOptions options = parseArgs(argc, argv);
        const RuntimeConfig& config = options.config;

        std::signal(SIGINT, onSignal);
        std::signal(SIGTERM, onSignal);

        std::filesystem::create_directories(config.output.directory);
        {
            std::ofstream out(config.output.directory + "/effective_config.json");
            out << configToJson(config) << '\n';
        }

        Profiler profiler;
        profiler.openOutputs(config.output.directory, config.output.save_frames_csv, config.output.save_jsonl);

        std::unique_ptr<Detector> detector = createDetector(config, profiler);
        std::unique_ptr<FrameSource> source = createSource(config);

        Pipeline pipeline(config, *detector, profiler);
        profiler.start();

        const uint64_t interval_ns = static_cast<uint64_t>(config.output.overlay_interval_ms)*1'000'000ULL;
        uint64_t last_print_ns = now_monotonic_ns();
        size_t last_print_frames = 0;

        auto on_frame = [&](const PipelineStatus& s){
            if(!config.output.overlay) return;
            const uint64_t now = now_monotonic_ns();
            if(now - last_print_ns < interval_ns) return;
            const double fps = (s.frames - last_print_frames) / ((now - last_print_ns)/1e9);
            printStatus(s, fps);
            last_print_ns = now;
            last_print_frames = s.frames;
        };

        pipeline.run(*source, config.pipeline.max_frames, &g_stop, on_frame);

        const DetectorWorker& worker = pipeline.worker();
        profiler.setExtra("worker_submitted", static_cast<double>(worker.submitted()));
        profiler.setExtra("worker_replaced", static_cast<double>(worker.replaced()));
        profiler.setExtra("worker_skipped", static_cast<double>(worker.skipped()));
        profiler.setExtra("worker_dropped_results", static_cast<double>(worker.droppedResults()));
        profiler.setExtra("tracker_stale_batches", static_cast<double>(pipeline.tracker().staleBatches()));
        profiler.setExtra("async", config.pipeline.async ? 1.0 : 0.0);
        profiler.setExtra("simulated_delay_ms", config.pipeline.simulated_delay_ms);

#ifdef EDGE_AI_HAVE_V4L2
        if(auto* camera = dynamic_cast<V4L2Source*>(source.get())){
            const V4L2Stats stats = camera->stats();
            profiler.setExtra("camera_captured", static_cast<double>(stats.captured));
            profiler.setExtra("camera_dropped", static_cast<double>(stats.dropped));
            profiler.setExtra("camera_effective_fps", stats.effective_fps);
        }
#endif

        const std::string summary_path = config.output.directory + "/summary.json";
        profiler.writeSummary(summary_path);

        const size_t frames = profiler.frames();
        const size_t calls = profiler.detectorCalls();
        std::cerr
            << "frames " << frames
            << " | detector calls " << calls
            << " | avoided " << (frames > calls ? 100.0*(frames - calls)/frames : 0.0) << "%"
            << "\nsummary: " << summary_path << '\n';
        return 0;
    }
    catch(const std::exception& error){
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
