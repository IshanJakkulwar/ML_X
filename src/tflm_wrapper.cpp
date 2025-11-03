// src/tflm_wrapper.cpp
#include "../include/ml_addon/tflm_wrapper.hpp"
#include <stdexcept>

namespace ml_addon {
namespace tflm {

ModelHandle* load_model_from_buffer(const unsigned char* /*data*/, size_t /*len*/) {
    // Stub: to use real TFLite Micro:
    // - include tflite-micro sources
    // - allocate arena buffer
    // - call tflite::GetModel, tflite::MicroInterpreter, etc.
    return nullptr;
}

std::vector<float> run_inference(ModelHandle* h, const std::vector<float>& /*input*/) {
    if (!h) throw std::runtime_error("TFLM model not loaded");
    // placeholder (should never be called in stub)
    return std::vector<float>(h->output_size, 0.0f);
}

void free_model(ModelHandle* /*h*/) {
    // free resources in real implementation
}

} // namespace tflm
} // namespace ml_addon
