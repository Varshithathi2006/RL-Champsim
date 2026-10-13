#include "hierarchical_rl_logger.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>

hierarchical_rl_logger::hierarchical_rl_logger(CACHE* cache) : hierarchical_rl_logger(cache, cache->NUM_SET, cache->NUM_WAY) {}

hierarchical_rl_logger::hierarchical_rl_logger(CACHE* cache, long sets, long ways)
    : replacement(cache),
      NUM_SET(sets),
      NUM_WAY(ways),
      last_used_cycles(static_cast<std::size_t>(sets * ways), 0),
      insertion_cycles(static_cast<std::size_t>(sets * ways), 0),
      access_counts(static_cast<std::size_t>(sets * ways), 0),
      block_addresses(static_cast<std::size_t>(sets * ways), 0),
      block_valid(static_cast<std::size_t>(sets * ways), 0),
      log_file(std::make_unique<std::ofstream>("hierarchical_rl_log.csv", std::ios::out))
{
  if (log_file && log_file->is_open()) {
    *log_file << "cycle,cpu,ip,full_addr,set,way,hit,type,rolling_miss_rate,rolling_read_ratio,recency_ranks,access_counts,insertion_ages\n";
  }
}

hierarchical_rl_logger::~hierarchical_rl_logger()
{
  if (log_file && log_file->is_open()) {
    log_file->close();
  }
}

long hierarchical_rl_logger::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                                         champsim::address full_addr, access_type type)
{
  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  // Default LRU victim choice
  auto victim = std::min_element(begin, end);
  assert(begin <= victim);
  assert(victim < end);
  return std::distance(begin, victim);
}

void hierarchical_rl_logger::replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                                    champsim::address victim_addr, access_type type)
{
  std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + way);
  last_used_cycles.at(idx) = cycle;
  insertion_cycles.at(idx) = cycle;
  access_counts.at(idx) = 1;
  block_addresses.at(idx) = full_addr.to<uint64_t>();
  block_valid.at(idx) = 1;
}

void hierarchical_rl_logger::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                                       champsim::address victim_addr, access_type type, uint8_t hit)
{
  cycle++;
  total_accesses++;
  if (!hit) total_misses++;

  // Update rolling phase statistics (1000-access window)
  rolling_access_outcomes.push_back(hit);
  if (!hit) rolling_miss_count++;
  
  bool is_read = (access_type{type} == access_type::LOAD || access_type{type} == access_type::TRANSLATION);
  rolling_access_types.push_back(is_read ? 1 : 0);
  if (is_read) rolling_read_count++;

  if (rolling_access_outcomes.size() > 1000) {
    if (rolling_access_outcomes.front() == 0) rolling_miss_count--;
    rolling_access_outcomes.pop_front();

    if (rolling_access_types.front() == 1) rolling_read_count--;
    rolling_access_types.pop_front();
  }

  double rolling_miss_rate = rolling_access_outcomes.empty() ? 0.0 : (double)rolling_miss_count / rolling_access_outcomes.size();
  double rolling_read_ratio = rolling_access_types.empty() ? 0.0 : (double)rolling_read_count / rolling_access_types.size();

  // Compute recency ranks (0 = MRU, NUM_WAY-1 = LRU)
  std::vector<uint64_t> set_cycles(NUM_WAY);
  std::vector<uint32_t> set_counts(NUM_WAY);
  std::vector<uint64_t> set_ages(NUM_WAY);

  for (long w = 0; w < NUM_WAY; ++w) {
    std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + w);
    set_cycles[w] = last_used_cycles.at(idx);
    set_counts[w] = access_counts.at(idx);
    set_ages[w] = (cycle >= insertion_cycles.at(idx)) ? (cycle - insertion_cycles.at(idx)) : 0;
  }

  std::string recency_str = "";
  std::string count_str = "";
  std::string age_str = "";

  for (long w = 0; w < NUM_WAY; ++w) {
    long rank = 0;
    for (long other = 0; other < NUM_WAY; ++other) {
      if (set_cycles[other] > set_cycles[w]) {
        rank++;
      }
    }
    recency_str += std::to_string(rank);
    count_str += std::to_string(set_counts[w]);
    age_str += std::to_string(set_ages[w]);

    if (w < NUM_WAY - 1) {
      recency_str += ";";
      count_str += ";";
      age_str += ";";
    }
  }

  // Log to CSV on every access (or misses)
  if (log_file && log_file->is_open()) {
    *log_file << cycle << ","
              << triggering_cpu << ","
              << "0x" << std::hex << ip.to<uint64_t>() << std::dec << ","
              << "0x" << std::hex << full_addr.to<uint64_t>() << std::dec << ","
              << set << ","
              << way << ","
              << static_cast<int>(hit) << ","
              << champsim::to_underlying(type) << ","
              << std::fixed << std::setprecision(4) << rolling_miss_rate << ","
              << std::fixed << std::setprecision(4) << rolling_read_ratio << ","
              << recency_str << ","
              << count_str << ","
              << age_str << "\n";
  }

  // Update per-block stats for hits
  if (hit && access_type{type} != access_type::WRITE) {
    std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + way);
    last_used_cycles.at(idx) = cycle;
    access_counts.at(idx)++;
    block_addresses.at(idx) = full_addr.to<uint64_t>();
    block_valid.at(idx) = 1;
  }
}

void hierarchical_rl_logger::replacement_final_stats()
{
  if (log_file && log_file->is_open()) {
    log_file->flush();
    log_file->close();
  }
}
