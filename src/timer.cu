#include "timer.hh"
#include <algorithm>
#include <iostream>
#include <vector>

std::chrono::high_resolution_clock::time_point start_timer() {
	return std::chrono::high_resolution_clock::now();
}

/*std::chrono::duration<double>*/ float duration(std::chrono::high_resolution_clock::time_point start) {
	auto end_time =std::chrono::high_resolution_clock::now(); 	
	return std::chrono::duration_cast<std::chrono::duration<float>>(end_time - start).count() * 1000;
}

std::pair<cudaEvent_t, cudaEvent_t>  start_timer_gpu()
{
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  return std::pair<cudaEvent_t, cudaEvent_t>(start, stop);
}

float duration_gpu(std::pair<cudaEvent_t, cudaEvent_t>& events)
{
  cudaEventRecord(events.second);
  cudaEventSynchronize(events.second);

  float duration = 0;
  cudaEventElapsedTime(&duration, events.first, events.second);

  return duration;
}

void display_times(std::chrono::duration<double> cpu_duration, std::chrono::duration<double> gpu_duration, const std::string& category) {
    std::cout << category << '\n';
    std::cout << "GPU excution time: " << gpu_duration.count() * 1000 << "ms\n\n";
    std::cout << "CPU execution time: " << cpu_duration.count() * 1000 << "ms\n\n";
}

void display_times(const std::vector<float>& durations,
                    const std::vector<std::string>& categories, const std::string& implementation) {
  if (durations.size() != categories.size()) {
    std::cerr << "Different sizes of durations and categories\n";
    return;
  }
  std::cout << "-----------" << implementation << "-----------\n";
  for (int i = 0; i < durations.size(); ++i) {
    std::cout << categories[i] << " : " << durations[i] << "ms\n";
  }
}

void compute_duration(std::function<void()> func, std::vector<float>& durations)
{
  auto start_gray_gpu = start_timer_gpu();
  func();
  durations.emplace_back(duration_gpu(start_gray_gpu));  
}
