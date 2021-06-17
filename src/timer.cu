#include "timer.hh"
#include <iostream>
#include <vector>

std::chrono::high_resolution_clock::time_point start_timer() {
	return std::chrono::high_resolution_clock::now();
}

std::chrono::duration<double> duration(std::chrono::high_resolution_clock::time_point start) {
	auto end_time =std::chrono::high_resolution_clock::now(); 	
	return end_time - start;
}

void display_times(std::chrono::duration<double> cpu_duration, std::chrono::duration<double> gpu_duration, const std::string& category) {
    std::cout << category << '\n';
    std::cout << "GPU excution time: " << gpu_duration.count() * 1000 << "ms\n\n";
    std::cout << "CPU execution time: " << cpu_duration.count() * 1000 << "ms\n\n";
}

void display_times(const std::vector<std::chrono::duration<double>>& durations,
                    const std::vector<std::string>& categories, const std::string& implementation) {
  if (durations.size() != categories.size()) {
    std::cerr << "Different sizes of durations and categories\n";
    return;
  }
  std::cout << "-----------" << implementation << "-----------\n";
  for (int i = 0; i < durations.size(); ++i) {
    std::cout << categories[i] << " : " << durations[i].count() * 1000 << "ms\n";
  }
}