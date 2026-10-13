#ifndef REPLACEMENT_RL_LOGGER_H
#define REPLACEMENT_RL_LOGGER_H

#include <memory>
#include <vector>
#include <fstream>
#include <string>
#include <cstdint>
#include "cache.h"
#include "modules.h"

class rl_logger : public champsim::modules::replacement
{
  long NUM_SET;
  long NUM_WAY;
  std::vector<uint64_t> last_used_cycles;
  std::vector<uint64_t> block_addresses;
  std::vector<uint8_t> block_valid;
  uint64_t cycle = 0;
  std::unique_ptr<std::ofstream> log_file;

  long get_belady_victim(long set, const champsim::cache_block* current_set);

public:
  rl_logger(CACHE* cache);
  rl_logger(CACHE* cache, long sets, long ways);
  rl_logger(rl_logger&&) = default;
  rl_logger& operator=(rl_logger&&) = default;
  ~rl_logger();

  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                   champsim::address full_addr, access_type type);
  void replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                              access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                access_type type, uint8_t hit);
  void replacement_final_stats();
};

#endif
