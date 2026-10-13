#include "hierarchical_rl.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

hierarchical_rl::hierarchical_rl(CACHE* cache) : hierarchical_rl(cache, cache->NUM_SET, cache->NUM_WAY) {}

hierarchical_rl::hierarchical_rl(CACHE* cache, long sets, long ways)
    : replacement(cache),
      NUM_SET(sets),
      NUM_WAY(ways),
      last_used_cycles(static_cast<std::size_t>(sets * ways), 0),
      insertion_cycles(static_cast<std::size_t>(sets * ways), 0),
      access_counts(static_cast<std::size_t>(sets * ways), 0),
      block_addresses(static_cast<std::size_t>(sets * ways), 0),
      block_valid(static_cast<std::size_t>(sets * ways), 0)
{
  weights_loaded = load_weights_json("hierarchical_dqn_weights.json");
  if (!weights_loaded) {
    std::cout << "[hierarchical_rl] Warning: Could not load 'hierarchical_dqn_weights.json'. Falling back to LRU.\n";
  } else {
    std::cout << "[hierarchical_rl] Successfully loaded 2-Level Hierarchical Neural Network weights for C++ inference.\n";
  }
}

bool hierarchical_rl::load_weights_json(const std::string& path)
{
  std::ifstream f(path);
  if (!f.is_open()) return false;

  try {
    json j = json::parse(f);

    high_agent_layers.clear();
    for (const auto& l : j["high_level_agent"]) {
      LayerWeights layer;
      layer.weight = l["weight"].get<std::vector<std::vector<float>>>();
      layer.bias = l["bias"].get<std::vector<float>>();
      high_agent_layers.push_back(layer);
    }

    low_agent_layers.clear();
    for (const auto& l : j["low_level_agent"]) {
      LayerWeights layer;
      layer.weight = l["weight"].get<std::vector<std::vector<float>>>();
      layer.bias = l["bias"].get<std::vector<float>>();
      low_agent_layers.push_back(layer);
    }

    return !high_agent_layers.empty() && !low_agent_layers.empty();
  } catch (...) {
    return false;
  }
}

std::vector<float> hierarchical_rl::forward_layer(const std::vector<float>& in, const LayerWeights& layer, bool relu)
{
  std::size_t out_dim = layer.weight.size();
  std::size_t in_dim = in.size();
  std::vector<float> out(out_dim, 0.0f);

  for (std::size_t i = 0; i < out_dim; ++i) {
    float sum = layer.bias[i];
    for (std::size_t j = 0; j < in_dim; ++j) {
      sum += layer.weight[i][j] * in[j];
    }
    if (relu && sum < 0.0f) sum = 0.0f;
    out[i] = sum;
  }

  return out;
}

long hierarchical_rl::predict_victim_way(long set, uint8_t type_val)
{
  double rolling_miss_rate = rolling_access_outcomes.empty() ? 0.5 : (double)rolling_miss_count / rolling_access_outcomes.size();
  double rolling_read_ratio = rolling_access_types.empty() ? 0.5 : (double)rolling_read_count / rolling_access_types.size();

  // 1. High-Level Forward Pass (Phase Features -> 16-dim Phase Embedding)
  std::vector<float> high_x = { (float)rolling_miss_rate, (float)rolling_read_ratio };
  std::vector<float> phase_embed = high_x;
  for (std::size_t i = 0; i < high_agent_layers.size(); ++i) {
    bool is_last = (i == high_agent_layers.size() - 1);
    phase_embed = forward_layer(phase_embed, high_agent_layers[i], true); // ReLU on hidden & output
  }

  // 2. Low-Level Way Features
  std::vector<uint64_t> set_cycles(NUM_WAY);
  std::vector<uint32_t> set_counts(NUM_WAY);
  std::vector<uint64_t> set_ages(NUM_WAY);

  for (long w = 0; w < NUM_WAY; ++w) {
    std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + w);
    set_cycles[w] = last_used_cycles.at(idx);
    set_counts[w] = access_counts.at(idx);
    set_ages[w] = (cycle >= insertion_cycles.at(idx)) ? (cycle - insertion_cycles.at(idx)) : 0;
  }

  std::vector<float> low_x;
  low_x.reserve(56);

  // Recency ranks
  for (long w = 0; w < NUM_WAY; ++w) {
    long rank = 0;
    for (long other = 0; other < NUM_WAY; ++other) {
      if (set_cycles[other] > set_cycles[w]) rank++;
    }
    low_x.push_back((float)rank / (NUM_WAY - 1));
  }

  // Frequency & Age
  for (long w = 0; w < NUM_WAY; ++w) low_x.push_back(std::min((float)set_counts[w] / 10.0f, 1.0f));
  for (long w = 0; w < NUM_WAY; ++w) low_x.push_back(std::min((float)std::log1p(set_ages[w]) / 15.0f, 1.0f));

  // Type One-Hot
  for (int t = 0; t < 8; ++t) low_x.push_back(t == (int)type_val ? 1.0f : 0.0f);

  // Concatenate Low-Level Features + High-Level Phase Embedding
  std::vector<float> combined_x = low_x;
  combined_x.insert(combined_x.end(), phase_embed.begin(), phase_embed.end());

  // 3. Low-Level Forward Pass
  std::vector<float> q_values = combined_x;
  for (std::size_t i = 0; i < low_agent_layers.size(); ++i) {
    bool is_last = (i == low_agent_layers.size() - 1);
    q_values = forward_layer(q_values, low_agent_layers[i], !is_last);
  }

  // Argmax Q-value
  auto max_it = std::max_element(q_values.begin(), q_values.end());
  long victim_way = std::distance(q_values.begin(), max_it);
  return (victim_way >= 0 && victim_way < NUM_WAY) ? victim_way : 0;
}

long hierarchical_rl::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                                  champsim::address full_addr, access_type type)
{
  if (weights_loaded) {
    return predict_victim_way(set, champsim::to_underlying(type));
  }

  // Fallback to LRU
  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);
  auto victim = std::min_element(begin, end);
  return std::distance(begin, victim);
}

void hierarchical_rl::replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                             champsim::address victim_addr, access_type type)
{
  std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + way);
  last_used_cycles.at(idx) = cycle;
  insertion_cycles.at(idx) = cycle;
  access_counts.at(idx) = 1;
  block_addresses.at(idx) = full_addr.to<uint64_t>();
  block_valid.at(idx) = 1;
}

void hierarchical_rl::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                                champsim::address victim_addr, access_type type, uint8_t hit)
{
  cycle++;
  total_accesses++;
  if (!hit) total_misses++;

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

  if (hit && access_type{type} != access_type::WRITE) {
    std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + way);
    last_used_cycles.at(idx) = cycle;
    access_counts.at(idx)++;
    block_addresses.at(idx) = full_addr.to<uint64_t>();
    block_valid.at(idx) = 1;
  }
}

void hierarchical_rl::replacement_final_stats()
{
  std::cout << "[hierarchical_rl] Final Stats: Total Accesses=" << total_accesses << ", Total Misses=" << total_misses << "\n";
}
