#include "softmax.h"
#include <algorithm>
#include <cmath>
#include <vector>


void softmax(const float *input, float *output, int size)
{
    float max_element = *std::max_element(input, input+size);
    float denom = 0.0f;
    for (int i=0; i<size; ++i){
        output[i] = std::exp(input[i]-max_element);
        denom += output[i];
    }
    for (int i=0; i<size; ++i){
        output[i] /= denom;
    }
}


void softmax_vector(const std::vector<float>& input, std::vector<float>& output){
    output.clear();
    float denom = .0f;
    auto max_element = *std::max_element(input.begin(), input.end());
    for(const auto& elem: input){
        auto exp_elem = std::exp(elem - max_element);
        output.push_back(exp_elem);
        denom += exp_elem;
    }

    for (auto &elem : output){
        elem /= denom;
    }
}