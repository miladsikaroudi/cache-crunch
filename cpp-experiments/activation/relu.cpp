#include "relu.h"

void relu(const std::vector<float> &input, std::vector<float>& output){
    output.clear();
    output.reserve(input.size());
    for (const auto& elem: input){
        output.push_back(elem>0 ? elem : 0);
    }

}