#pragma once
#include <vector>
void layer_norm(const std::vector<float> &input, std::vector<float> &output, float epsilon = 1e-5f);
void layer_norm_raw (const float * input, float* output, int size, float epsilon = 1e-5f);