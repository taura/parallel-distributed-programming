#pragma once

#include <chrono>
#include <vector>

void LayerNorm(const std::vector<float>& x, int rows, int cols, const std::vector<float>& w,
               const std::vector<float>& b, float eps, std::vector<float>& y);
void Linear(const std::vector<float>& x, int rows, int in_dim, int out_dim,
            const std::vector<float>& w, const std::vector<float>& b, std::vector<float>& y);
float Gelu(float x);
void ApplyGelu(std::vector<float>& x);
void AddInPlace(std::vector<float>& dst, const std::vector<float>& src);
bool HasNanOrInf(const std::vector<float>& x);
int Argmax(const std::vector<float>& x);
double ChecksumPrefix(const std::vector<float>& x, int n);
double ElapsedMs(std::chrono::steady_clock::time_point t0,
                 std::chrono::steady_clock::time_point t1);
