#include <cmath>
#include "normalization.h"

void layer_norm(const std::vector<float> &input, std::vector<float> &output, float epsilon)
{
    output.clear();
    output.reserve(input.size());
    auto mean = .0f;
    for (const auto& elem: input){
        mean += elem;
    }
    mean /= input.size();

    auto variance = .0f;
    for (const auto & elem: input){
        variance += (elem - mean) * (elem - mean);
    }
    variance /= input.size();

    for (auto const & elem : input){

        output.push_back((elem - mean) / (std::sqrt(variance+epsilon)));
    }
}

void layer_norm_raw(const float *input, float *output, int size, float epsilon)
{
    float mean = .0f;
    for (int i=0; i<size; ++i){
        mean += input[i];
    }
    mean /= size;

    float var = .0f;
    for (int i=0; i<size; ++i){
        var += std::pow((input[i]-mean), 2);
    }
    var /= size;

    for (int i=0; i<size; ++i){
        output[i] = (input[i]-mean)/(std::sqrt(var+epsilon));
    }
}