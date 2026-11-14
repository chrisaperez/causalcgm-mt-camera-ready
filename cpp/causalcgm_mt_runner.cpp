// CausalCGM-MT C++ inference runner (ONNX Runtime).
//
// Purpose: establish whether a 30-minute-ahead forecast plus hypoglycemia risk
// flag can be produced inside a real-time medical-device budget (< 50 ms).
//
// Methodology notes, because latency benchmarks are easy to do dishonestly:
//
//   * We report the full percentile distribution, not the mean. For a device
//     the number that matters is the tail -- a p50 of 2 ms with a p99.9 of
//     80 ms misses the deadline several times an hour at 5-minute cadence.
//   * Warm-up iterations are discarded. The first inference pays for lazy
//     kernel selection and arena allocation and is not representative.
//   * Timing wraps ONLY session.Run(). Tensor construction from a caller-owned
//     buffer is measured separately, since in a real device the feature buffer
//     is filled incrementally as samples arrive, not built per inference.
//   * The default is single-threaded. A device is not going to hand four cores
//     to the forecaster, and letting ORT spin up a thread pool flatters the
//     numbers relative to the deployment target.
//
// The exported graph folds in the population scaler and the output sigmoid, so
// this runner takes RAW features and needs no preprocessing constants of its
// own -- there is no way for it to drift out of sync with training.
//
// Build: see cpp/CMakeLists.txt

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Defaults only. The real geometry is READ OFF THE GRAPH in `GraphSpec` below,
// because hard-coding it is how the runner came to be unable to serve the
// model the paper actually proposes: `kChannels = 5` and two bound inputs
// silently restricted it to the CGM-only architecture, while the fusion model
// needs `x_aux` and `aux_present` as well.
constexpr int64_t kDefaultHistorySteps = 24;   // H = 120 min @ 5 min
constexpr int64_t kCgmChannels = 5;   // value, delta, roll_mean, roll_std, mask
constexpr double kBudgetMs = 50.0;    // real-time device constraint

// What the loaded graph actually expects. Populated by inspecting the session,
// so a graph with a different history length, a different auxiliary width or a
// different input set is served correctly -- or rejected with a clear message
// -- rather than being fed a tensor of the wrong shape.
struct GraphSpec {
  std::vector<std::string> input_names;
  int64_t history_steps = kDefaultHistorySteps;
  int64_t cgm_channels = kCgmChannels;
  int64_t aux_channels = 0;      // 0 when the graph has no x_aux input
  bool multimodal = false;       // graph binds x_aux / aux_present

  bool Has(const std::string& n) const {
    return std::find(input_names.begin(), input_names.end(), n) !=
           input_names.end();
  }
};

GraphSpec InspectGraph(Ort::Session* session) {
  GraphSpec spec;
  Ort::AllocatorWithDefaultOptions alloc;
  const size_t n_in = session->GetInputCount();

  for (size_t i = 0; i < n_in; ++i) {
    auto name = session->GetInputNameAllocated(i, alloc);
    const std::string n(name.get());
    spec.input_names.push_back(n);

    const auto shape = session->GetInputTypeInfo(i)
                           .GetTensorTypeAndShapeInfo()
                           .GetShape();
    if (n == "x_raw" && shape.size() == 3) {
      spec.history_steps = shape[1];
      spec.cgm_channels = shape[2];
    } else if (n == "x_aux" && shape.size() == 3) {
      spec.aux_channels = shape[2];
    }
  }

  spec.multimodal = spec.Has("x_aux");
  if (spec.multimodal && !spec.Has("aux_present")) {
    throw std::runtime_error(
        "graph binds x_aux but not aux_present; the fusion model needs both");
  }
  if (!spec.Has("x_raw") || !spec.Has("last_bg")) {
    throw std::runtime_error("graph must expose x_raw and last_bg inputs");
  }
  if (spec.cgm_channels != kCgmChannels) {
    throw std::runtime_error(
        "x_raw has " + std::to_string(spec.cgm_channels) +
        " channels; BuildFeatures produces " + std::to_string(kCgmChannels));
  }
  return spec;
}

struct Percentiles {
  double min, p50, p90, p99, p999, max, mean;
};

Percentiles Summarize(std::vector<double> xs) {
  std::sort(xs.begin(), xs.end());
  auto at = [&](double q) {
    if (xs.empty()) return 0.0;
    const size_t i = static_cast<size_t>(q * (xs.size() - 1));
    return xs[i];
  };
  const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
  return {xs.front(), at(0.50), at(0.90), at(0.99), at(0.999), xs.back(),
          sum / static_cast<double>(xs.size())};
}

// Default short-gap budget, in grid steps. Mirrors
// `DataConfig.max_interpolation_gap_min / sample_period_min` = 15 / 5. It is a
// property of the TRAINED pipeline rather than of the device, so the verify
// mode takes it as an argument and `scripts/verify_cpp_parity.py` passes the
// value out of the checkpoint's own config; this constant is only the fallback.
constexpr int64_t kDefaultMaxGapSteps = 3;

// Mirrors causalcgm_mt.data._causal_gap_fill, and it is NOT optional.
//
// In training, `build_windows` fills a window's short interior gaps BEFORE
// featurising, so the model has never once seen a raw value at an unobserved
// step. The runner used to featurise whatever it was handed, which made a
// routine CGM dropout a train/serve skew: measured against the Python pipeline
// on a 10-minute interior gap, features diverged by up to 1.3e3 mg/dL and the
// forecast by up to 1.8e2 mg/dL, while every graph-level parity check still
// passed -- because both sides of those checks are the graph.
//
// Interior gaps of at most `max_gap_steps` are linearly interpolated; both
// endpoints lie inside the window, so the fill is causal with respect to the
// anchor. Leading and trailing unobserved steps are deliberately NOT filled:
// doing so needs a value from outside the window (leading) or an extrapolation
// (trailing).
//
// Returns false when some step is left neither observed nor imputed -- a gap
// longer than the budget, or an unfilled edge. `build_windows` DROPS such a
// window, so the model was never trained on one and the caller must not ask for
// a forecast from it.
bool CausalGapFill(std::vector<float>* values,
                   const std::vector<float>& observed,
                   int64_t max_gap_steps,
                   std::vector<uint8_t>* imputed) {
  const int64_t n = static_cast<int64_t>(values->size());
  imputed->assign(static_cast<size_t>(n), 0);

  std::vector<int64_t> seen;
  for (int64_t i = 0; i < n; ++i) {
    if (observed[static_cast<size_t>(i)] != 0.0f) seen.push_back(i);
  }
  if (seen.empty()) return false;

  for (size_t k = 0; k + 1 < seen.size(); ++k) {
    const int64_t a = seen[k], b = seen[k + 1];
    const int64_t gap = b - a - 1;
    if (gap <= 0 || gap > max_gap_steps) continue;
    // Interpolated in double and stored as float, so the fill itself does not
    // spend any of the 1e-3 mg/dL parity budget.
    const double va = (*values)[static_cast<size_t>(a)];
    const double vb = (*values)[static_cast<size_t>(b)];
    for (int64_t j = a + 1; j < b; ++j) {
      const double w = static_cast<double>(j - a) / static_cast<double>(b - a);
      (*values)[static_cast<size_t>(j)] = static_cast<float>(va + w * (vb - va));
      (*imputed)[static_cast<size_t>(j)] = 1;
    }
  }

  for (int64_t i = 0; i < n; ++i) {
    if (observed[static_cast<size_t>(i)] == 0.0f &&
        !(*imputed)[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

// Build one causal feature window from a raw glucose history.
//
// This mirrors causalcgm_mt.data.causal_features EXACTLY, including the rule that
// feature at index i depends only on history[0..i]. Any divergence here is a
// train/serve skew bug that no amount of ONNX parity checking would catch,
// because the parity check compares graphs, not the code that feeds them.
void BuildFeatures(const std::vector<float>& history,  // length H
                   const std::vector<float>& observed,
                   std::vector<float>* out /* H * C, row-major */) {
  const int64_t kHistorySteps = static_cast<int64_t>(history.size());
  constexpr int64_t kChannels = kCgmChannels;
  out->assign(kHistorySteps * kChannels, 0.0f);
  constexpr int kRollWindow = 6;

  for (int64_t i = 0; i < kHistorySteps; ++i) {
    const float v = history[i];
    const float delta = (i == 0) ? 0.0f : history[i] - history[i - 1];

    const int64_t lo = std::max<int64_t>(0, i - kRollWindow + 1);
    const int64_t n = i - lo + 1;

    float mean = 0.0f;
    for (int64_t j = lo; j <= i; ++j) mean += history[j];
    mean /= static_cast<float>(n);

    float var = 0.0f;
    if (n > 1) {
      for (int64_t j = lo; j <= i; ++j) var += (history[j] - mean) * (history[j] - mean);
      var /= static_cast<float>(n);  // population std, matching numpy default
    }

    float* row = out->data() + i * kChannels;
    row[0] = v;
    row[1] = delta;
    row[2] = mean;
    row[3] = std::sqrt(var);
    row[4] = observed[i];
  }
}

}  // namespace

// Verification mode: read a glucose history from stdin (one value per line),
// run it, and print the outputs at full precision.
//
// This exists to close the one gap the ONNX parity check cannot cover. That
// check compares two graphs; it says nothing about whether BuildFeatures above
// agrees with causalcgm_mt.data.causal_features. A divergence there is classic
// train/serve skew -- the deployed model would receive subtly different
// features from the ones it was trained on, and every graph-level test would
// still pass. scripts/verify_cpp_parity.py drives this mode and diffs the
// result against the Python pipeline.
int RunVerify(Ort::Session* session, const GraphSpec& spec,
              int64_t max_gap_steps) {
  // stdin: H glucose values, optionally followed by H observation flags. The
  // mask half is optional so existing callers keep working, and it exists
  // because without it the parity claim only ever covered fully-observed
  // windows -- the mask channel was hard-coded to ones on both sides, so a
  // divergence in how partial observation is encoded could not have been seen.
  std::vector<float> raw;
  float v;
  while (std::cin >> v) raw.push_back(v);

  const int64_t H = spec.history_steps;
  std::vector<float> history, observed;
  if (static_cast<int64_t>(raw.size()) == H) {
    history.assign(raw.begin(), raw.end());
    observed.assign(H, 1.0f);
  } else if (static_cast<int64_t>(raw.size()) == 2 * H) {
    history.assign(raw.begin(), raw.begin() + H);
    observed.assign(raw.begin() + H, raw.end());
  } else {
    std::cerr << "expected " << H << " values (or " << 2 * H
              << " with an observation mask), got " << raw.size() << "\n";
    return 1;
  }

  // Fill the window's short interior gaps BEFORE featurising -- the order the
  // training pipeline uses. The observation mask written into channel 4 stays
  // the RAW one: `build_windows` records what the device reported, not what was
  // reconstructed, so an imputed step is honestly marked unobserved.
  std::vector<uint8_t> imputed;
  if (!CausalGapFill(&history, observed, max_gap_steps, &imputed)) {
    std::cerr << "window is not usable: after filling interior gaps of at most "
              << max_gap_steps << " step(s), some step is still unobserved. "
                 "build_windows drops such a window, so the model was never "
                 "trained on one -- do not forecast from it.\n";
    return 2;
  }

  std::vector<float> window;
  BuildFeatures(history, observed, &window);

  const std::array<int64_t, 3> x_shape{1, H, spec.cgm_channels};
  const std::array<int64_t, 1> scalar_shape{1};
  // The FILLED anchor value, matching `build_windows`' `lbg.append(hv[-1])`.
  std::vector<float> last_bg{history.back()};

  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Bind by NAME in the graph's own order, so a graph with extra inputs is
  // served rather than silently mis-fed.
  std::vector<float> x_aux(spec.multimodal ? H * spec.aux_channels : 0, 0.0f);
  std::vector<float> aux_present{0.0f};   // absent -> exact CGM-only path
  const std::array<int64_t, 3> aux_shape{1, H, spec.aux_channels};

  std::vector<const char*> in_names;
  std::vector<Ort::Value> inputs;
  for (const auto& n : spec.input_names) {
    in_names.push_back(n.c_str());
    if (n == "x_raw") {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem, window.data(), window.size(), x_shape.data(), x_shape.size()));
    } else if (n == "last_bg") {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem, last_bg.data(), last_bg.size(), scalar_shape.data(),
          scalar_shape.size()));
    } else if (n == "x_aux") {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem, x_aux.data(), x_aux.size(), aux_shape.data(), aux_shape.size()));
    } else if (n == "aux_present") {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem, aux_present.data(), aux_present.size(), scalar_shape.data(),
          scalar_shape.size()));
    } else {
      std::cerr << "unknown graph input: " << n << "\n";
      return 1;
    }
  }

  const char* output_names[] = {"bg_hat", "risk"};
  auto out = session->Run(Ort::RunOptions{nullptr}, in_names.data(),
                          inputs.data(), inputs.size(), output_names, 2);

  // Emit the features too, so a mismatch can be localised to the feature code
  // rather than merely detected at the output.
  std::cout << std::setprecision(9);
  std::cout << "features\n";
  for (int64_t i = 0; i < H; ++i) {
    for (int64_t c = 0; c < spec.cgm_channels; ++c) {
      std::cout << window[i * spec.cgm_channels + c]
                << (c + 1 == spec.cgm_channels ? '\n' : ' ');
    }
  }
  // With K > 1 the outputs are (1, K); print every horizon so the Python side
  // can diff whichever it scored.
  const auto n_out = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
  std::cout << "bg_hat";
  for (size_t k = 0; k < n_out; ++k) std::cout << ' ' << out[0].GetTensorData<float>()[k];
  std::cout << "\nrisk";
  for (size_t k = 0; k < n_out; ++k) std::cout << ' ' << out[1].GetTensorData<float>()[k];
  std::cout << "\n";
  return 0;
}

int main(int argc, char** argv) {
  const bool verify = (argc > 1) && std::string(argv[1]) == "--verify";
  const std::string model_path =
      verify ? (argc > 2 ? argv[2] : "runs/demo/causalcgm_mt.onnx")
             : ((argc > 1) ? argv[1] : "runs/demo/causalcgm_mt.onnx");
  // Positional benchmark args are only meaningful outside --verify mode, where
  // argv[2] is the model path rather than an iteration count.
  // In verify mode argv[3] is the short-gap budget in steps; see
  // `kDefaultMaxGapSteps`. Outside it, the positional args stay what they were.
  const int64_t max_gap_steps =
      (verify && argc > 3) ? std::stoll(argv[3]) : kDefaultMaxGapSteps;
  const int iterations = (!verify && argc > 2) ? std::stoi(argv[2]) : 2000;
  const int warmup = (!verify && argc > 3) ? std::stoi(argv[3]) : 200;
  const int64_t batch = (!verify && argc > 4) ? std::stoll(argv[4]) : 1;
  const int threads = (!verify && argc > 5) ? std::stoi(argv[5]) : 1;

  if (verify) {
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "causalcgm_mt_verify");
    Ort::SessionOptions o;
    o.SetIntraOpNumThreads(1);
    Ort::Session s(env, model_path.c_str(), o);
    try {
      return RunVerify(&s, InspectGraph(&s), max_gap_steps);
    } catch (const std::exception& e) {
      std::cerr << "graph inspection failed: " << e.what() << "\n";
      return 1;
    }
  }

  std::cout << "CausalCGM-MT ONNX Runtime benchmark\n"
            << "  model      : " << model_path << "\n"
            << "  batch      : " << batch << "\n"
            << "  threads    : " << threads << "\n"
            << "  iterations : " << iterations << " (+" << warmup << " warmup)\n"
            << "  ORT        : " << OrtGetApiBase()->GetVersionString() << "\n\n";

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "causalcgm_mt");
  Ort::SessionOptions opts;
  // Single-threaded and fully optimised: the realistic device configuration.
  opts.SetIntraOpNumThreads(threads);
  opts.SetInterOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  std::unique_ptr<Ort::Session> session;
  try {
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);
  } catch (const Ort::Exception& e) {
    std::cerr << "Failed to load model: " << e.what() << "\n"
              << "Export it first:  python scripts/export_onnx.py\n";
    return 1;
  }

  GraphSpec spec;
  try {
    spec = InspectGraph(session.get());
  } catch (const std::exception& e) {
    std::cerr << "graph inspection failed: " << e.what() << "\n";
    return 1;
  }
  const int64_t kHistorySteps = spec.history_steps;
  const int64_t kChannels = spec.cgm_channels;

  std::cout << "  graph      : " << (spec.multimodal ? "multimodal fusion"
                                                     : "CGM-only")
            << ", H=" << kHistorySteps << ", cgm_ch=" << kChannels;
  if (spec.multimodal) std::cout << ", aux_ch=" << spec.aux_channels;
  std::cout << "\n\n";

  // --- synthetic but realistic input ---------------------------------------
  std::mt19937 rng(42);
  std::normal_distribution<float> noise(0.0f, 6.0f);

  std::vector<float> history(kHistorySteps), observed(kHistorySteps, 1.0f);
  for (int64_t i = 0; i < kHistorySteps; ++i) {
    history[i] = 130.0f + 25.0f * std::sin(static_cast<float>(i) * 0.28f) + noise(rng);
  }

  std::vector<float> window;
  const auto feat_t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; ++i) BuildFeatures(history, observed, &window);
  const auto feat_t1 = std::chrono::steady_clock::now();
  const double feat_us =
      std::chrono::duration<double, std::micro>(feat_t1 - feat_t0).count() / 1000.0;

  // Replicate the single window across the batch.
  std::vector<float> x(batch * kHistorySteps * kChannels);
  for (int64_t b = 0; b < batch; ++b) {
    std::memcpy(x.data() + b * kHistorySteps * kChannels, window.data(),
                window.size() * sizeof(float));
  }
  std::vector<float> last_bg(batch, history.back());

  const std::array<int64_t, 3> x_shape{batch, kHistorySteps, kChannels};
  const std::array<int64_t, 1> scalar_shape{batch};
  const std::array<int64_t, 3> aux_shape{batch, kHistorySteps, spec.aux_channels};

  // Auxiliary stream: present and non-trivial, so the fusion branch, the
  // cross-attention and the gate are all exercised. Timing the fusion graph
  // with aux_present = 0 would measure the CGM-only path through a bigger
  // graph, which is the measurement error this change exists to fix.
  std::vector<float> x_aux(spec.multimodal ? batch * kHistorySteps * spec.aux_channels
                                           : 0);
  for (auto& a : x_aux) a = noise(rng) * 0.1f;
  std::vector<float> aux_present(spec.multimodal ? batch : 0, 1.0f);

  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const char* output_names[] = {"bg_hat", "risk"};

  // Tensors wrap caller-owned memory, so they are constructed once and the
  // buffer is updated in place -- which is how a streaming device would do it.
  // Bound by NAME in the graph's own order.
  std::vector<const char*> input_names;
  std::vector<Ort::Value> inputs;
  for (const auto& n : spec.input_names) {
    input_names.push_back(n.c_str());
    if (n == "x_raw") {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem, x.data(), x.size(), x_shape.data(), x_shape.size()));
    } else if (n == "last_bg") {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem, last_bg.data(), last_bg.size(), scalar_shape.data(),
          scalar_shape.size()));
    } else if (n == "x_aux") {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem, x_aux.data(), x_aux.size(), aux_shape.data(), aux_shape.size()));
    } else if (n == "aux_present") {
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem, aux_present.data(), aux_present.size(), scalar_shape.data(),
          scalar_shape.size()));
    } else {
      std::cerr << "unknown graph input: " << n << "\n";
      return 1;
    }
  }

  // --- warm-up (discarded) --------------------------------------------------
  for (int i = 0; i < warmup; ++i) {
    session->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                 inputs.size(), output_names, 2);
  }

  // --- timed loop -----------------------------------------------------------
  std::vector<double> ms;
  ms.reserve(iterations);
  float last_bg_out = 0.0f, last_risk = 0.0f;

  for (int i = 0; i < iterations; ++i) {
    // Perturb the input each iteration so nothing can be cached across runs.
    x[(kHistorySteps - 1) * kChannels] = history.back() + noise(rng) * 0.01f;

    const auto t0 = std::chrono::steady_clock::now();
    auto out = session->Run(Ort::RunOptions{nullptr}, input_names.data(),
                            inputs.data(), inputs.size(), output_names, 2);
    const auto t1 = std::chrono::steady_clock::now();

    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    last_bg_out = out[0].GetTensorData<float>()[0];
    last_risk = out[1].GetTensorData<float>()[0];
  }

  const Percentiles p = Summarize(ms);
  const double per_window_p999 = p.p999 / static_cast<double>(batch);

  std::cout << std::fixed << std::setprecision(3)
            << "Feature construction : " << feat_us << " us / window\n\n"
            << "Inference latency (ms, per Run call)\n"
            << "  min    " << p.min << "\n"
            << "  mean   " << p.mean << "\n"
            << "  p50    " << p.p50 << "\n"
            << "  p90    " << p.p90 << "\n"
            << "  p99    " << p.p99 << "\n"
            << "  p99.9  " << p.p999 << "\n"
            << "  max    " << p.max << "\n\n"
            << "Per-window p99.9     : " << per_window_p999 << " ms\n"
            << "Throughput (p50)     : "
            << (static_cast<double>(batch) / p.p50 * 1000.0) << " windows/s\n\n"
            << "Sample output        : bg_hat " << last_bg_out
            << " mg/dL, risk " << last_risk << "\n\n";

  // The budget is judged on the TAIL, not the average. A device that misses
  // its deadline one time in a thousand at 5-minute cadence misses it roughly
  // every three days.
  const bool ok = p.max <= kBudgetMs;
  std::cout << "Budget " << kBudgetMs << " ms : "
            << (ok ? "PASS" : "FAIL") << "  (worst observed " << p.max << " ms, "
            << std::setprecision(1) << (p.max / kBudgetMs * 100.0)
            << "% of budget)\n";
  return ok ? 0 : 2;
}
