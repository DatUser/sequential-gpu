#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <functional>

std::chrono::high_resolution_clock::time_point start_timer();
/*std::chrono::duration<double>*/float duration(std::chrono::high_resolution_clock::time_point start);
void display_times(std::chrono::duration<double> cpu_duration, std::chrono::duration<double> gpu_duration, const std::string& category);
void display_times(std::chrono::duration<double> cpu_duration, std::chrono::duration<double> gpu_duration_1,
                   std::chrono::duration<double> gpu_duration_2, const std::string& category);

std::pair<cudaEvent_t, cudaEvent_t>  start_timer_gpu();
float duration_gpu(std::pair<cudaEvent_t, cudaEvent_t>& events);


void display_times(const std::vector<float>& durations,
                   const std::vector<std::string>& categories, const std::string& implementation);

void compute_duration(std::function<void()> func, std::vector<float>& durations);
