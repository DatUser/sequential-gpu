#include "timer.hh"
#include <iostream>

std::chrono::high_resolution_clock::time_point start_timer() {
	return std::chrono::high_resolution_clock::now();
}

std::chrono::duration<double> duration(std::chrono::high_resolution_clock::time_point start) {
	auto end_time =std::chrono::high_resolution_clock::now(); 	
	return end_time - start;
}

void display_times(std::chrono::duration<double> cpu_duration, std::chrono::duration<double> gpu_duration, const std::string& category) {
    std::cout << category << '\n';
    std::cout << "GPU excution time:\n" << gpu_duration.count() * 1000 << "ms\n\n";
    std::cout << "CPU execution time:\n" << cpu_duration.count() * 1000 << "ms\n";
}