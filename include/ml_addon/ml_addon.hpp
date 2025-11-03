#pragma once
// ml_addon.hpp
// Minimal ML addon for PROS V5: tiny MLP inference, feature scaler, online linear learner.
// Drop into include/ml_addon/

#include <vector>
#include <cstddef>
#include <cstdint>

namespace ml_addon {

// Simple feature scaler supporting two modes:
// - "minmax": x' = (x - min) / (max - min)
// - "standard": x' = (x - mean) / std
struct Scaler {
    enum Mode { MINMAX = 0, STANDARD = 1 };
    Mode mode = MINMAX;
    std::vector<float> param1; // min or mean
    std::vector<float> param2; // max or std
    float clip_min = -10.0f;
    float clip_max = 10.0f;

    // load stats (param vectors must match input dim)
    void load(const std::vector<float>& p1, const std::vector<float>& p2, Mode m);
    std::vector<float> transform(const std::vector<float>& in) const;
};

// Small fully-connected MLP inference engine.
// Weights stored as vector<out_dim * in_dim> per layer (row-major).
class MLP {
public:
    // load precomputed arrays (from C header exported by train_export.py)
    bool load_from_arrays(const std::vector<std::vector<float>>& W,
                          const std::vector<std::vector<float>>& B,
                          const std::vector<size_t>& sizes,
                          const std::vector<const char*>& activations);

    // infer (throws on mismatch)
    std::vector<float> infer(const std::vector<float>& input) const;

    size_t input_size() const;
    size_t output_size() const;

private:
    std::vector<size_t> layer_sizes_;
    std::vector<const char*> activations_;
    std::vector<std::vector<float>> W_;
    std::vector<std::vector<float>> B_;
};

// Tiny online linear model (y = W*x + b) trained with SGD on the device.
// Useful to adapt a small residualer or for online PID tuning suggestions.
class OnlineLinear {
public:
    OnlineLinear() = default;
    // initialize dims
    void init(size_t input_dim, size_t output_dim, float lr = 1e-3f, float l2 = 0.0f);
    // predict
    std::vector<float> predict(const std::vector<float>& x) const;
    // single-step SGD update: minimize MSE between pred and target
    void update(const std::vector<float>& x, const std::vector<float>& target);
    // load/save to arrays
    void load(const std::vector<float>& W_flat, const std::vector<float>& b_flat, size_t in_dim, size_t out_dim);
    void export_params(std::vector<float>& W_flat, std::vector<float>& b_flat) const;

private:
    size_t in_dim_ = 0;
    size_t out_dim_ = 0;
    float lr_ = 1e-3f;
    float l2_ = 0.0f;
    std::vector<float> W_; // out_dim * in_dim (row-major: out * in + in)
    std::vector<float> b_; // out_dim
};

// Circular buffer (small) for streaming feature windows
template<typename T>
class CircularBuffer {
public:
    CircularBuffer() = default;
    CircularBuffer(size_t capacity) { init(capacity); }
    void init(size_t capacity);
    void push(const T& v);
    // get contiguous vector of last n elements (n <= size)
    std::vector<T> last_n(size_t n) const;
    size_t size() const { return size_; }
    size_t capacity() const { return cap_; }
private:
    std::vector<T> data_;
    size_t head_ = 0;
    size_t size_ = 0;
    size_t cap_ = 0;
};

// Exponential weighted moving average helper
class EWMA {
public:
    explicit EWMA(float alpha = 0.1f) : alpha_(alpha) {}
    void reset(float v = 0.0f) { value_ = v; initialized_ = true; }
    float update(float v);
    float value() const { return value_; }
private:
    float alpha_;
    float value_ = 0.0f;
    bool initialized_ = false;
};

} // namespace ml_addon
