#ifndef REPLACEMENT_HIERARCHICAL_RL_H
#define REPLACEMENT_HIERARCHICAL_RL_H

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

class hierarchical_rl : public champsim::modules::replacement
{
  long NUM_SET;
  long NUM_WAY;

  std::vector<uint64_t> last_used_cycles;
  std::vector<uint64_t> insertion_cycles;
  std::vector<uint32_t> access_counts;
  std::vector<uint64_t> block_addresses;
  std::vector<uint8_t> block_valid;

  uint64_t cycle = 0;
  uint64_t total_accesses = 0;
  uint64_t total_misses = 0;

  // Rolling phase statistics (1000 accesses window)
  std::deque<uint8_t> rolling_access_outcomes;
  std::deque<uint8_t> rolling_access_types;
  uint32_t rolling_miss_count = 0;
  uint32_t rolling_read_count = 0;

  // Neural Network Weights loaded from JSON
  bool weights_loaded = false;
  std::vector<LayerWeights> high_agent_layers;
  std::vector<LayerWeights> low_agent_layers;

  bool load_weights_json(const std::string& path);
  std::vector<float> forward_layer(const std::vector<float>& in, const LayerWeights& layer, bool relu = true);
  long predict_victim_way(long set, uint8_t type_val);

public:
  hierarchical_rl(CACHE* cache);
  hierarchical_rl(CACHE* cache, long sets, long ways);
  hierarchical_rl(hierarchical_rl&&) = default;
  hierarchical_rl& operator=(hierarchical_rl&&) = default;
  ~hierarchical_rl() = default;

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                   champsim::address full_addr, access_type type);
  void replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                              access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                access_type type, uint8_t hit);
  void replacement_final_stats();
};

#endif
