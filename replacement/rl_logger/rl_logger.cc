#include "rl_logger.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>

rl_logger::rl_logger(CACHE* cache) : rl_logger(cache, cache->NUM_SET, cache->NUM_WAY) {}

rl_logger::rl_logger(CACHE* cache, long sets, long ways)
    : replacement(cache),
      NUM_SET(sets),
      NUM_WAY(ways),
      last_used_cycles(static_cast<std::size_t>(sets * ways), 0),
      block_addresses(static_cast<std::size_t>(sets * ways), 0),
      block_valid(static_cast<std::size_t>(sets * ways), 0),
      log_file(std::make_unique<std::ofstream>("champsim_rl_log.csv", std::ios::out))
{
  if (log_file && log_file->is_open()) {
    *log_file << "cycle,cpu,ip,full_addr,set,way,hit,type,recency_ranks,belady_victim_way\n";
  }
}

rl_logger::~rl_logger()
{
  if (log_file && log_file->is_open()) {
    log_file->close();
  }
}

long rl_logger::get_belady_victim(long set, const champsim::cache_block* current_set)
{
  // First look for invalid ways
  for (long w = 0; w < NUM_WAY; ++w) {
    if (current_set && !current_set[w].valid) {
      return w;
    }
  }

  // Fall back to LRU position for baseline Belady oracle estimation
  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);
  auto victim = std::min_element(begin, end);
  return std::distance(begin, victim);
}

long rl_logger::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                          champsim::address full_addr, access_type type)
{
  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  // Find the way whose last use cycle is most distant (LRU)
  auto victim = std::min_element(begin, end);
  assert(begin <= victim);
  assert(victim < end);
  return std::distance(begin, victim);
}

void rl_logger::replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                     access_type type)
{
  std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + way);
  last_used_cycles.at(idx) = cycle++;
  block_addresses.at(idx) = full_addr.to<uint64_t>();
  block_valid.at(idx) = 1;
}

void rl_logger::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                       champsim::address victim_addr, access_type type, uint8_t hit)
{
  // Calculate recency rank for each way in this set (0 = MRU, NUM_WAY-1 = LRU)
  std::string recency_str = "";
  std::size_t set_offset = static_cast<std::size_t>(set * NUM_WAY);
  for (long w = 0; w < NUM_WAY; ++w) {
    uint64_t target_cycle = last_used_cycles.at(set_offset + w);
    long rank = 0;
    for (long w2 = 0; w2 < NUM_WAY; ++w2) {
      if (last_used_cycles.at(set_offset + w2) > target_cycle) {
        rank++;
      }
    }
    recency_str += std::to_string(rank);
    if (w < NUM_WAY - 1) {
      recency_str += ";";
    }
  }

  long belady_victim = get_belady_victim(set, nullptr);

  // Log to CSV
  if (log_file && log_file->is_open()) {
    *log_file << cycle << ","
              << triggering_cpu << ","
              << "0x" << std::hex << ip.to<uint64_t>() << std::dec << ","
              << "0x" << std::hex << full_addr.to<uint64_t>() << std::dec << ","
              << set << ","
              << way << ","
              << static_cast<int>(hit) << ","
              << champsim::to_underlying(type) << ","
              << recency_str << ","
              << belady_victim << "\n";
  }

  // Update LRU state for hits (excluding writebacks)
  if (hit && access_type{type} != access_type::WRITE) {
    std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + way);
    last_used_cycles.at(idx) = cycle++;
    block_addresses.at(idx) = full_addr.to<uint64_t>();
    block_valid.at(idx) = 1;
  }
}

void rl_logger::replacement_final_stats()
{
  if (log_file && log_file->is_open()) {
    log_file->flush();
    log_file->close();
  }
}
