#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "image.hh"
#include "save.hh"
#include "blocks-gpu.hh"
#include "canonical-gpu.hh"
#include "image-gpu.hh"
#include "tests.hh"
#include "timer.hh"

std::vector<std::chrono::duration<double>> gpu_canonical(int window_size) {

  std::vector<std::chrono::duration<double>> durations;
  ImageGPU img_gpu("data/test.jpg");

  auto start_gray_gpu = start_timer();
  img_gpu.to_gray();
  durations.emplace_back(duration(start_gray_gpu));

  auto start_padd_gpu = start_timer();
  img_gpu.padd_image();
  durations.emplace_back(duration(start_padd_gpu));

  auto start_to_blocks_canonical_gpu = start_timer();
  CanonicalGPU canonical_gpu(img_gpu, window_size);
  durations.emplace_back(duration(start_to_blocks_canonical_gpu));

  auto start_texton_gpu = start_timer();
  canonical_gpu.compute_textons();
  durations.emplace_back(duration(start_texton_gpu));

  auto start_histo_gpu = start_timer();
  //blocks_gpu.compute_histogram_blocks();
  canonical_gpu.compute_shared_histogram_blocks();
  durations.emplace_back(duration(start_histo_gpu));

  return durations;
}

std::vector<std::chrono::duration<double>> gpu_blocks(int window_size, bool histogram_shared = false) {

  std::vector<std::chrono::duration<double>> durations;
  ImageGPU img_gpu("data/test.jpg");

  auto start_gray_gpu = start_timer();
  img_gpu.to_gray();
  durations.emplace_back(duration(start_gray_gpu));

  auto start_padd_gpu = start_timer();
  img_gpu.padd_image();
  durations.emplace_back(duration(start_padd_gpu));

  auto start_to_blocks_canonical_gpu = start_timer();
  BlocksGPU blocks_gpu = img_gpu.to_blocks(window_size);
  durations.emplace_back(duration(start_to_blocks_canonical_gpu));

  auto start_texton_gpu = start_timer();
  blocks_gpu.compute_textons();
  durations.emplace_back(duration(start_texton_gpu));

  if (histogram_shared) {
    auto start_histo_gpu = start_timer();
    blocks_gpu.compute_histogram_blocks();
    durations.emplace_back(duration(start_histo_gpu));
  }
  else {
    auto start_histo_gpu = start_timer();
    blocks_gpu.compute_histogram_blocks();
    durations.emplace_back(duration(start_histo_gpu));
  }

  return durations;
}

std::vector<std::chrono::duration<double>> cpu_implementation(int window_size, bool histogram_shared = false) {
  std::vector<std::chrono::duration<double>> durations;
  Image img_cpu("data/test.jpg");
  int patch_size = 16;
  img_cpu.set_patch_size(patch_size);

  auto start_gray_cpu = start_timer();
  auto other_img = img_cpu.to_gray();
  durations.emplace_back(duration(start_gray_cpu));


  auto start_padd_cpu = start_timer();
  auto padded_img = other_img.add_padding_row();
  auto padded_img2 = padded_img.add_padding_col();
  durations.emplace_back(duration(start_padd_cpu));

  auto start_to_blocks_cpu = start_timer();
  Blocks blocks = padded_img2.to_blocks(window_size);
  durations.emplace_back(duration(start_to_blocks_cpu));

  auto start_texton_cpu = start_timer();
  blocks.compute_textons_blocks();
  durations.emplace_back(duration(start_texton_cpu));

  auto start_histo_cpu = start_timer();
  blocks.compute_histogram_blocks();
  std::vector<int> hist = blocks.get_concatenated_histograms();
  durations.emplace_back(duration(start_histo_cpu));

  return durations;
}

void test(int window_size, bool histogram_shared = false) {
  // CANONICAL
  ImageGPU img_gpu_canonical("data/test.jpg");
  img_gpu_canonical.to_gray();
  img_gpu_canonical.padd_image();
  CanonicalGPU canonical_gpu(img_gpu_canonical, window_size);
  canonical_gpu.compute_textons();

  canonical_gpu.compute_shared_histogram_blocks();

  ImageGPU img_test("data/test.jpg");
  img_test.set_padded_gray_data(canonical_gpu.textons_device);
  BlocksGPU blocks_gpu_test = img_test.to_blocks(window_size);

  //BLOCKS
  ImageGPU img_gpu("data/test.jpg");
  img_gpu.to_gray();
  img_gpu.padd_image();
  BlocksGPU blocks_gpu = img_gpu.to_blocks(window_size);
  blocks_gpu.compute_textons();

  std::cout << blocks_gpu.get_concatenated_histograms_size() << " " <<  blocks_gpu_test.get_concatenated_histograms_size() << "\n";
  std::cout << blocks_gpu.textons_device[0];
  std::cout << "Accessed blocks  texton first element" << std::endl;
  std::cout << blocks_gpu_test.blocks_device[0];
  std::cout << "Accessed canonical texton first element" << std::endl;



  bool are_texton_eq = are_array_equal<unsigned char *>(blocks_gpu.textons_device, blocks_gpu_test.blocks_device,
                                                       blocks_gpu.get_concatenated_histograms_size(), blocks_gpu_test.get_concatenated_histograms_size());
  std::cout << "--------------\n";
  std::cout << "Canonical : Texton test: " << std::boolalpha << are_texton_eq << '\n';


  if (histogram_shared)
    blocks_gpu.compute_histogram_blocks();
  else
    blocks_gpu.compute_histogram_blocks();
  //CPU
  Image img_cpu("data/test.jpg");
  int patch_size = 16;
  img_cpu.set_patch_size(patch_size);
  auto other_img = img_cpu.to_gray();
  auto padded_img = other_img.add_padding_row();
  auto padded_img2 = padded_img.add_padding_col();
  Blocks blocks = padded_img2.to_blocks(window_size);
  blocks.compute_textons_blocks();
  blocks.compute_histogram_blocks();
    
  std::vector<int> canonical_gpu_histogram_vector(canonical_gpu.histogram, canonical_gpu.histogram + canonical_gpu.get_concatenated_histograms_size());
  save_csv("data/gpu_canonical_histogram.csv", ",", canonical_gpu_histogram_vector, 256);

  std::vector<int> blocks_gpu_histogram_vector(blocks_gpu.histogram, blocks_gpu.histogram + blocks_gpu.get_concatenated_histograms_size());
  save_csv("data/gpu_blocks_histogram.csv", ",", blocks_gpu_histogram_vector, 256);

  std::vector<int> hist = blocks.get_concatenated_histograms();
  save_csv("data/histogram.csv", ",", hist, patch_size * patch_size);

  bool are_histo_eq_canonical = are_array_equal<int *>(blocks.get_concatenated_histograms().data(), canonical_gpu.histogram,
                                             blocks.get_concatenated_histograms_size(), canonical_gpu.get_concatenated_histograms_size());
  std::cout << "--------------\n";
 std::cout << "Canonical : Concatenate test: " << std::boolalpha << are_histo_eq_canonical << '\n';

  bool are_histo_eq = are_array_equal<int *>(blocks.get_concatenated_histograms().data(), blocks_gpu.histogram,
                                             blocks.get_concatenated_histograms_size(), blocks_gpu.get_concatenated_histograms_size());
  std::cout << "--------------\n";
  std::cout << "Block : Concatenate test: " << std::boolalpha << are_histo_eq << '\n';

  std::vector<Block*> blocks_data = blocks.get_blocks();
  //Block* data = blocks_data[0];
  //std::cout << *data << '\n';

}

/*bool are_eq1 = are_array_equal<unsigned char *>(other_img.get_data(), img_gpu.get_gray_data(), other_img.get_size(), img_gpu.get_size());
bool are_eq2 = are_array_equal<unsigned char *>(padded_img2.get_data(), img_gpu.get_padded_gray_data(), padded_img2.get_size(), img_gpu.get_padded_size());
std::cout << "--------------\n";
std::cout << "Gray Img test: " << std::boolalpha << are_eq1 << '\n';
std::cout << "Padded Gray Img test: " << std::boolalpha << are_eq2 << '\n';*/

int main() {
  std::vector<std::string> categories = {"Gray", "Pad", "To Blocks", "Texton", "Histo"};
  int window_size = 3;
  test(window_size);
  auto durations_canonical = gpu_canonical(window_size);
  auto durations_blocks = gpu_blocks(window_size, false);
  auto duration_cpu = cpu_implementation(window_size);
  display_times(duration_cpu, categories, "CPU");
  display_times(durations_blocks, categories, "GPU BLOCKS");
  display_times(durations_canonical, categories, "GPU CANONICAL");

  return 0;
}
