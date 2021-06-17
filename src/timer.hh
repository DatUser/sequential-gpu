#pragma once
#include <chrono>
#include <string>
#include <vector>

std::chrono::high_resolution_clock::time_point start_timer();
std::chrono::duration<double> duration(std::chrono::high_resolution_clock::time_point start);
void display_times(std::chrono::duration<double> cpu_duration, std::chrono::duration<double> gpu_duration, const std::string& category);
void display_times(std::chrono::duration<double> cpu_duration, std::chrono::duration<double> gpu_duration_1,
                   std::chrono::duration<double> gpu_duration_2, const std::string& category);
void display_times(const std::vector<std::chrono::duration<double>>& durations,
                   const std::vector<std::string>& categories, const std::string& implementation);
