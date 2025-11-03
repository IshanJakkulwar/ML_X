#pragma once
// Minimal TFLite Micro wrapper header (interface only).
// This file provides a clean API so you can add TensorFlow Lite Micro
// without changing your application code. The implementation here
// is a stub that compiles without TFLite; to enable TFLite, implement
// the functions using TFLite Micro library and link it.

#include <vector>
#include <cstddef>

namespace ml_addon {
namespace tflm {

// Opaque model handle
struct ModelHandle { size_t input_size = 0; size_t output_size = 0; };

// Load a TFLite flatbuffer from memory (pointer + length).
// In the stub, this always returns nullptr. With TFLM, return a real handle.
ModelHandle* load_model_from_buffer(const unsigned char* data, size_t len);

// Run inference (input length must match model input). Returns output vector (size = output_size).
std::vector<float> run_inference(ModelHandle* h, const std::vector<float>& input);

// Free model handle
void free_model(ModelHandle* h);

} // namespace tflm
} // namespace ml_addon
