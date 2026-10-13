#ifndef REPLACEMENT_PHASE_AWARE_RL_H
#define REPLACEMENT_PHASE_AWARE_RL_H

#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <deque>
#include "cache.h"
#include "modules.h"

struct LayerWeights {
  std::vector<std::vector<float>> weight; // [out_features, in_features]
  std::vector<float> bias;                 // [out_features]
};

struct AccessRecord {
  uint64_t cpu_cycle;
  uint64_t full_addr;
  uint8_t hit;
  uint8_t type;
};

class phase_aware_rl : public champsim::modules::replacement
{
  long NUM_SET;
  long NUM_WAY;

  std::vector<uint64_t> last_used_cycles;
  std::vector<uint64_t> insertion_cycles;
  std::vector<uint32_t> access_counts;

  uint64_t total_accesses = 0;
  uint64_t total_misses = 0;

  // Window size N = 1000
  static constexpr std::size_t WINDOW_SIZE = 1000;
  std::deque<AccessRecord> access_window;

  // Rolling counters
  uint32_t window_misses = 0;
  uint32_t window_writes = 0;
  uint32_t window_streaming_count = 0;
  uint64_t prev_block_addr = 0;

  // Neural Network Weights & KMeans Model
  bool weights_loaded = false;
  std::vector<LayerWeights> dqn_layers;

  std::vector<float> scaler_mean;
  std::vector<float> scaler_scale;
  std::vector<std::vector<float>> cluster_centers;

  bool load_models();
  std::vector<float> forward_layer(const std::vector<float>& in, const LayerWeights& layer, bool relu = true);
  int predict_phase_cluster(float miss_rate, float write_frac, float reuse_frac, float access_rate);
  long predict_victim_way(long set, access_type type, uint64_t current_cpu_cycle);

public:
  phase_aware_rl(CACHE* cache);
  phase_aware_rl(CACHE* cache, long sets, long ways);
  phase_aware_rl(phase_aware_rl&&) = default;
  phase_aware_rl& operator=(phase_aware_rl&&) = default;
  ~phase_aware_rl() = default;

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                   champsim::address full_addr, access_type type);
  void replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                              access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                access_type type, uint8_t hit);
  void replacement_final_stats();
};

#endif
