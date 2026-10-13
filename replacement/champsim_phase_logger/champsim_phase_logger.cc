#include "champsim_phase_logger.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <unordered_map>

champsim_phase_logger::champsim_phase_logger(CACHE* cache) : champsim_phase_logger(cache, cache->NUM_SET, cache->NUM_WAY) {}

champsim_phase_logger::champsim_phase_logger(CACHE* cache, long sets, long ways)
    : replacement(cache),
      NUM_SET(sets),
      NUM_WAY(ways),
      last_used_cycles(static_cast<std::size_t>(sets * ways), 0),
      insertion_cycles(static_cast<std::size_t>(sets * ways), 0),
      access_counts(static_cast<std::size_t>(sets * ways), 0)
{
  phase_log_file.open("champsim_phase_log.csv");
  if (phase_log_file.is_open()) {
    phase_log_file << "cycle,cpu,full_addr,set,way,hit,type,recency_ranks,rolling_miss_rate,write_fraction,streaming_fraction,reuse_fraction,no_reuse_fraction,avg_reuse_distance,llc_access_rate\n";
  }
}

champsim_phase_logger::~champsim_phase_logger()
{
  if (phase_log_file.is_open()) {
    phase_log_file.close();
  }
}

long champsim_phase_logger::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                                         champsim::address full_addr, access_type type)
{
  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);
  auto victim = std::min_element(begin, end);
  return std::distance(begin, victim);
}

void champsim_phase_logger::replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                                    champsim::address victim_addr, access_type type)
{
  uint64_t current_cpu_cycle = 0;
  if (intern_ && intern_->clock_period.count() > 0) {
    current_cpu_cycle = static_cast<uint64_t>(intern_->current_time.time_since_epoch() / intern_->clock_period);
  }
  std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + way);
  last_used_cycles.at(idx) = current_cpu_cycle;
  insertion_cycles.at(idx) = current_cpu_cycle;
  access_counts.at(idx) = 1;
}

void champsim_phase_logger::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                                       champsim::address victim_addr, access_type type, uint8_t hit)
{
  total_accesses++;
  if (!hit) total_misses++;

  uint64_t current_cpu_cycle = 0;
  if (intern_ && intern_->clock_period.count() > 0) {
    current_cpu_cycle = static_cast<uint64_t>(intern_->current_time.time_since_epoch() / intern_->clock_period);
  } else {
    current_cpu_cycle = total_accesses;
  }

  uint64_t addr_val = full_addr.to<uint64_t>();
  uint8_t type_val = champsim::to_underlying(type);
  bool is_write = (access_type{type} == access_type::WRITE);
  bool is_streaming = (prev_block_addr != 0 && (addr_val == prev_block_addr + 64 || addr_val == prev_block_addr + 4096));
  prev_block_addr = addr_val;

  AccessRecord rec{ current_cpu_cycle, addr_val, hit, type_val };
  access_window.push_back(rec);

  if (!hit) window_misses++;
  if (is_write) window_writes++;
  if (is_streaming) window_streaming_count++;

  if (access_window.size() > WINDOW_SIZE) {
    const auto& oldest = access_window.front();
    if (oldest.hit == 0) window_misses--;
    if (access_type{oldest.type} == access_type::WRITE) window_writes--;
    
    if (access_window.size() >= 2) {
      uint64_t a0 = access_window[0].full_addr;
      uint64_t a1 = access_window[1].full_addr;
      if (a1 == a0 + 64 || a1 == a0 + 4096) {
        if (window_streaming_count > 0) window_streaming_count--;
      }
    }
    access_window.pop_front();
  }

  if (hit && !is_write) {
    std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + way);
    last_used_cycles.at(idx) = current_cpu_cycle;
    access_counts.at(idx)++;
  }

  compute_and_log_window_stats(triggering_cpu, set, way, full_addr, type, hit, current_cpu_cycle);
}

void champsim_phase_logger::compute_and_log_window_stats(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, access_type type, uint8_t hit, uint64_t current_cpu_cycle)
{
  if (!phase_log_file.is_open()) return;

  // Compute recency ranks (0 = MRU, NUM_WAY-1 = LRU)
  std::vector<uint64_t> set_cycles(NUM_WAY);
  for (long w = 0; w < NUM_WAY; ++w) {
    std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + w);
    set_cycles[w] = last_used_cycles.at(idx);
  }

  std::string recency_str = "";
  for (long w = 0; w < NUM_WAY; ++w) {
    long rank = 0;
    for (long other = 0; other < NUM_WAY; ++other) {
      if (set_cycles[other] > set_cycles[w]) {
        rank++;
      }
    }
    recency_str += std::to_string(rank);
    if (w < NUM_WAY - 1) recency_str += ";";
  }

  std::size_t sz = access_window.size();
  double rolling_miss_rate = (sz > 0) ? (double)window_misses / sz : 0.0;
  double write_fraction = (sz > 0) ? (double)window_writes / sz : 0.0;
  double streaming_fraction = (sz > 0) ? (double)window_streaming_count / sz : 0.0;

  // Genuine LLC Access Rate (accesses per elapsed CPU cycles)
  uint64_t elapsed_cpu_cycles = (sz > 0 && current_cpu_cycle >= access_window.front().cpu_cycle) 
                                ? (current_cpu_cycle - access_window.front().cpu_cycle + 1) 
                                : 1;
  double llc_access_rate = (double)sz / elapsed_cpu_cycles;

  // Compute reuse fraction and average reuse distance ONLY over genuine reuses
  std::unordered_map<uint64_t, std::size_t> last_seen_pos;
  double total_reuse_dist = 0.0;
  std::size_t reuses = 0;

  for (std::size_t i = 0; i < sz; ++i) {
    uint64_t a = access_window[i].full_addr;
    if (last_seen_pos.count(a)) {
      total_reuse_dist += (i - last_seen_pos[a]);
      reuses++;
    }
    last_seen_pos[a] = i;
  }

  double reuse_fraction = (sz > 0) ? (double)reuses / sz : 0.0;
  double no_reuse_fraction = 1.0 - reuse_fraction;
  double avg_reuse_distance = (reuses > 0) ? (total_reuse_dist / reuses) : 0.0;

  phase_log_file << current_cpu_cycle << ","
                 << triggering_cpu << ","
                 << "0x" << std::hex << full_addr.to<uint64_t>() << std::dec << ","
                 << set << ","
                 << way << ","
                 << (int)hit << ","
                 << (int)champsim::to_underlying(type) << ","
                 << recency_str << ","
                 << rolling_miss_rate << ","
                 << write_fraction << ","
                 << streaming_fraction << ","
                 << reuse_fraction << ","
                 << no_reuse_fraction << ","
                 << avg_reuse_distance << ","
                 << llc_access_rate << "\n";
}

void champsim_phase_logger::replacement_final_stats()
{
  std::cout << "[champsim_phase_logger] Final Stats: Accesses=" << total_accesses << ", Misses=" << total_misses << "\n";
}
