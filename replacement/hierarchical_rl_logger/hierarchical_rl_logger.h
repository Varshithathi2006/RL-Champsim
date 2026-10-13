#ifndef REPLACEMENT_HIERARCHICAL_RL_LOGGER_H
#define REPLACEMENT_HIERARCHICAL_RL_LOGGER_H

#include <vector>
#include <fstream>
#include <string>
#include <cstdint>
#include <memory>
#include <deque>
#include "cache.h"
#include "modules.h"

class hierarchical_rl_logger : public champsim::modules::replacement
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

  // Rolling phase window (last 1000 accesses)
  std::deque<uint8_t> rolling_access_outcomes; // 1 = hit, 0 = miss
  std::deque<uint8_t> rolling_access_types;    // access_type enum values
  uint32_t rolling_miss_count = 0;
  uint32_t rolling_read_count = 0;

  std::unique_ptr<std::ofstream> log_file;

public:
  hierarchical_rl_logger(CACHE* cache);
  hierarchical_rl_logger(CACHE* cache, long sets, long ways);
  hierarchical_rl_logger(hierarchical_rl_logger&&) = default;
  hierarchical_rl_logger& operator=(hierarchical_rl_logger&&) = default;
  ~hierarchical_rl_logger();

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                   champsim::address full_addr, access_type type);
  void replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                              access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                access_type type, uint8_t hit);
  void replacement_final_stats();
};

#endif
