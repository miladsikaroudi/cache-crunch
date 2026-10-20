#pragma once
#include <vector>
void softmax(const float *input, float *output, int size);
void softmax_vector(const std::vector<float>& input, std::vector<float>& output);
