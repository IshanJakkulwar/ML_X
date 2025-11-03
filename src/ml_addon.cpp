// src/ml_addon.cpp
#include "../include/ml_addon/ml_addon.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstring>

namespace ml_addon {

// -------- Scaler ----------
void Scaler::load(const std::vector<float>& p1, const std::vector<float>& p2, Mode m) {
    param1 = p1; param2 = p2; mode = m;
}
std::vector<float> Scaler::transform(const std::vector<float>& in) const {
    if (in.size() != param1.size() || in.size() != param2.size()) throw std::runtime_error("Scaler size mismatch");
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        float v = in[i];
        if (mode == MINMAX) {
            float denom = (param2[i] - param1[i]);
            if (fabs(denom) < 1e-6f) out[i] = 0.0f;
            else out[i] = (v - param1[i]) / denom;
        } else {
            float stdv = param2[i];
            if (fabs(stdv) < 1e-6f) out[i] = 0.0f;
            else out[i] = (v - param1[i]) / stdv;
        }
        // clip
        if (out[i] < clip_min) out[i] = clip_min;
        if (out[i] > clip_max) out[i] = clip_max;
    }
    return out;
}

// -------- MLP -----------
bool MLP::load_from_arrays(const std::vector<std::vector<float>>& W,
                           const std::vector<std::vector<float>>& B,
                           const std::vector<size_t>& sizes,
                           const std::vector<const char*>& activations) {
    if (W.size() != sizes.size() - 1) return false;
    if (B.size() != sizes.size() - 1) return false;
    if (activations.size() != sizes.size() - 1) return false;
    W_ = W; B_ = B; layer_sizes_ = sizes; activations_ = activations;
    // basic dimension checks
    for (size_t l = 0; l < W_.size(); ++l) {
        size_t in = layer_sizes_[l];
        size_t out = layer_sizes_[l+1];
        if (W_[l].size() != out * in) return false;
        if (B_[l].size() != out) return false;
    }
    return true;
}
std::vector<float> MLP::infer(const std::vector<float>& input) const {
    if (input.size() != layer_sizes_.front()) throw std::runtime_error("MLP input mismatch");
    std::vector<float> cur = input;
    for (size_t l = 0; l < W_.size(); ++l) {
        size_t in = layer_sizes_[l];
        size_t out = layer_sizes_[l+1];
        std::vector<float> next(out, 0.0f);
        const auto& Wl = W_[l];
        const auto& Bl = B_[l];
        const char* act = activations_[l];
        for (size_t i = 0; i < out; ++i) {
            float acc = Bl[i];
            const float* wrow = &Wl[i * in];
            for (size_t j = 0; j < in; ++j) acc += wrow[j] * cur[j];
            if (std::strcmp(act, "relu") == 0) acc = acc > 0.0f ? acc : 0.0f;
            else if (std::strcmp(act, "sigmoid") == 0) acc = 1.0f / (1.0f + std::exp(-acc));
            else if (std::strcmp(act, "tanh") == 0) acc = std::tanh(acc);
            next[i] = acc;
        }
        cur.swap(next);
    }
    return cur;
}
size_t MLP::input_size() const { return layer_sizes_.empty() ? 0 : layer_sizes_.front(); }
size_t MLP::output_size() const { return layer_sizes_.empty() ? 0 : layer_sizes_.back(); }

// ----- OnlineLinear ----
void OnlineLinear::init(size_t input_dim, size_t output_dim, float lr, float l2) {
    in_dim_ = input_dim; out_dim_ = output_dim; lr_ = lr; l2_ = l2;
    W_.assign(out_dim_ * in_dim_, 0.0f);
    b_.assign(out_dim_, 0.0f);
}
std::vector<float> OnlineLinear::predict(const std::vector<float>& x) const {
    if (x.size() != in_dim_) throw std::runtime_error("OnlineLinear predict size mismatch");
    std::vector<float> y(out_dim_, 0.0f);
    for (size_t i = 0; i < out_dim_; ++i) {
        float acc = b_[i];
        const float* wrow = &W_[i * in_dim_];
        for (size_t j = 0; j < in_dim_; ++j) acc += wrow[j] * x[j];
        y[i] = acc;
    }
    return y;
}
void OnlineLinear::update(const std::vector<float>& x, const std::vector<float>& target) {
    if (x.size() != in_dim_ || target.size() != out_dim_) throw std::runtime_error("OnlineLinear update size mismatch");
    // compute prediction
    std::vector<float> y = predict(x);
    // simple SGD update
    for (size_t i = 0; i < out_dim_; ++i) {
        float err = y[i] - target[i];
        // bias update
        b_[i] -= lr_ * err;
        float* wrow = &W_[i * in_dim_];
        for (size_t j = 0; j < in_dim_; ++j) {
            // gradient = err * x_j + l2 * w
            float grad = err * x[j] + l2_ * wrow[j];
            wrow[j] -= lr_ * grad;
        }
    }
}
void OnlineLinear::load(const std::vector<float>& W_flat, const std::vector<float>& b_flat, size_t in_dim, size_t out_dim) {
    in_dim_ = in_dim; out_dim_ = out_dim;
    if (W_flat.size() != out_dim * in_dim) throw std::runtime_error("OnlineLinear load size mismatch");
    if (b_flat.size() != out_dim) throw std::runtime_error("OnlineLinear load size mismatch");
    W_ = W_flat; b_ = b_flat;
}
void OnlineLinear::export_params(std::vector<float>& W_flat, std::vector<float>& b_flat) const {
    W_flat = W_; b_flat = b_;
}

// ----- CircularBuffer ----
template<typename T>
void CircularBuffer<T>::init(size_t capacity) {
    data_.assign(capacity, T());
    head_ = 0; size_ = 0; cap_ = capacity;
}
template<typename T>
void CircularBuffer<T>::push(const T& v) {
    if (cap_ == 0) return;
    head_ = (head_ + 1) % cap_;
    data_[head_] = v;
    if (size_ < cap_) ++size_;
}
template<typename T>
std::vector<T> CircularBuffer<T>::last_n(size_t n) const {
    if (n == 0) return {};
    if (n > size_) n = size_;
    std::vector<T> out(n);
    size_t idx = head_;
    for (size_t k = 0; k < n; ++k) {
        out[n - 1 - k] = data_[idx];
        idx = (idx + cap_ - 1) % cap_;
    }
    return out;
}

// explicit instantiation for float and int32_t (common)
template class CircularBuffer<float>;
template class CircularBuffer<int32_t>;

// ---- EWMA ----
float EWMA::update(float v) {
    if (!initialized_) { value_ = v; initialized_ = true; return value_; }
    value_ = alpha_ * v + (1.0f - alpha_) * value_;
    return value_;
}

} // namespace ml_addon
