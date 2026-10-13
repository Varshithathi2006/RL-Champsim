#ifndef REPLACEMENT_CHAMPSIM_PHASE_LOGGER_H
#define REPLACEMENT_CHAMPSIM_PHASE_LOGGER_H

#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <deque>
#include <fstream>
#include <unordered_map>
#include "cache.h"
#include "modules.h"

struct AccessRecord {
  uint64_t cpu_cycle;
  uint64_t full_addr;
  uint8_t hit;
  uint8_t type; // LOAD=0, RFO=1, PREFETCH=2, WRITE=3, TRANSLATION=4
};

class champsim_phase_logger : public champsim::modules::replacement
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

  std::ofstream phase_log_file;

  void compute_and_log_window_stats(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, access_type type, uint8_t hit, uint64_t current_cpu_cycle);

public:
  champsim_phase_logger(CACHE* cache);
  champsim_phase_logger(CACHE* cache, long sets, long ways);
  champsim_phase_logger(champsim_phase_logger&&) = default;
  champsim_phase_logger& operator=(champsim_phase_logger&&) = default;
  ~champsim_phase_logger();

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                   champsim::address full_addr, access_type type);
  void replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                              access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                access_type type, uint8_t hit);
  void replacement_final_stats();
};

#endif
