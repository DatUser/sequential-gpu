#pragma once
#include <chrono>
#include <string>

std::chrono::high_resolution_clock::time_point start_timer();
std::chrono::duration<double> duration(std::chrono::high_resolution_clock::time_point start);
void display_times(std::chrono::duration<double> cpu_duration, std::chrono::duration<double> gpu_duration, const std::string& category);