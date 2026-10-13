#include "phase_aware_rl.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <unordered_map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

phase_aware_rl::phase_aware_rl(CACHE* cache) : phase_aware_rl(cache, cache->NUM_SET, cache->NUM_WAY) {}

phase_aware_rl::phase_aware_rl(CACHE* cache, long sets, long ways)
    : replacement(cache),
      NUM_SET(sets),
      NUM_WAY(ways),
      last_used_cycles(static_cast<std::size_t>(sets * ways), 0),
      insertion_cycles(static_cast<std::size_t>(sets * ways), 0),
      access_counts(static_cast<std::size_t>(sets * ways), 0)
{
  for (long s = 0; s < sets; ++s) {
    for (long w = 0; w < ways; ++w) {
      last_used_cycles[s * ways + w] = static_cast<uint64_t>(w + 1);
    }
  }
  weights_loaded = load_models();
  std::cout << "\n=======================================================\n";
  std::cout << ">>> [PHASE_AWARE_RL] POLICY INSTANTIATED AND ACTIVE <<<\n";
  if (!weights_loaded) {
    std::cout << "[phase_aware_rl] Warning: Could not load JSON weights. Falling back to LRU.\n";
  } else {
    std::cout << "[phase_aware_rl] Successfully loaded 28-dim Phase-Aware DQN and KMeans models for C++ inference.\n";
  }
  std::cout << "=======================================================\n\n";
}

bool phase_aware_rl::load_models()
{
  try {
    // 1. Load DQN Weights
    std::ifstream f_dqn("phase_aware_dqn_weights.json");
    if (!f_dqn.is_open()) return false;
    json j_dqn = json::parse(f_dqn);

    dqn_layers.clear();
    for (const auto& l : j_dqn) {
      LayerWeights layer;
      layer.weight = l["weight"].get<std::vector<std::vector<float>>>();
      layer.bias = l["bias"].get<std::vector<float>>();
      dqn_layers.push_back(layer);
    }

    // 2. Load KMeans & Scaler Model
    std::ifstream f_km("kmeans_phase_model.json");
    if (!f_km.is_open()) return false;
    json j_km = json::parse(f_km);

    scaler_mean = j_km["scaler_mean"].get<std::vector<float>>();
    scaler_scale = j_km["scaler_scale"].get<std::vector<float>>();
    cluster_centers = j_km["cluster_centers"].get<std::vector<std::vector<float>>>();

    return !dqn_layers.empty() && !cluster_centers.empty();
  } catch (...) {
    return false;
  }
}

std::vector<float> phase_aware_rl::forward_layer(const std::vector<float>& in, const LayerWeights& layer, bool relu)
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

int phase_aware_rl::predict_phase_cluster(float miss_rate, float write_frac, float reuse_frac, float access_rate)
{
  std::vector<float> raw = { miss_rate, write_frac, reuse_frac, access_rate };
  std::vector<float> scaled(4);
  for (int i = 0; i < 4; ++i) {
    float scale = (scaler_scale[i] > 1e-6f) ? scaler_scale[i] : 1.0f;
    scaled[i] = (raw[i] - scaler_mean[i]) / scale;
  }

  int best_cluster = 0;
  float min_dist = 1e9f;

  for (std::size_t c = 0; c < cluster_centers.size(); ++c) {
    float dist = 0.0f;
    for (int i = 0; i < 4; ++i) {
      float diff = scaled[i] - cluster_centers[c][i];
      dist += diff * diff;
    }
    if (dist < min_dist) {
      min_dist = dist;
      best_cluster = static_cast<int>(c);
    }
  }

  return best_cluster;
}

long phase_aware_rl::predict_victim_way(long set, access_type type, uint64_t current_cpu_cycle)
{
  std::size_t sz = access_window.size();
  float rolling_miss_rate = (sz > 0) ? (float)window_misses / sz : 0.0f;
  float write_fraction = (sz > 0) ? (float)window_writes / sz : 0.0f;
  float streaming_fraction = (sz > 0) ? (float)window_streaming_count / sz : 0.0f;

  uint64_t elapsed_cpu_cycles = (sz > 0 && current_cpu_cycle >= access_window.front().cpu_cycle) 
                                ? (current_cpu_cycle - access_window.front().cpu_cycle + 1) 
                                : 1;
  float llc_access_rate = (float)sz / elapsed_cpu_cycles;

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

  float reuse_fraction = (sz > 0) ? (float)reuses / sz : 0.0f;

  // 1. Predict 4-dim one-hot phase cluster
  int phase_cluster = predict_phase_cluster(rolling_miss_rate, write_fraction, reuse_fraction, llc_access_rate);
  std::vector<float> phase_onehot(4, 0.0f);
  if (phase_cluster >= 0 && phase_cluster < 4) {
    phase_onehot[phase_cluster] = 1.0f;
  }

  // 2. Build 16 Recency Ranks
  std::vector<uint64_t> set_cycles(NUM_WAY);
  for (long w = 0; w < NUM_WAY; ++w) {
    std::size_t idx = static_cast<std::size_t>(set * NUM_WAY + w);
    set_cycles[w] = last_used_cycles.at(idx);
  }

  std::vector<float> recency_norm;
  recency_norm.reserve(NUM_WAY);
  for (long w = 0; w < NUM_WAY; ++w) {
    long rank = 0;
    for (long other = 0; other < NUM_WAY; ++other) {
      if (set_cycles[other] > set_cycles[w]) rank++;
    }
    recency_norm.push_back((float)rank / (NUM_WAY - 1));
  }

  // 3. Build 8-dim Access Type One-Hot
  uint8_t type_val = champsim::to_underlying(type);
  std::vector<float> type_onehot(8, 0.0f);
  if (type_val < 8) {
    type_onehot[type_val] = 1.0f;
  }

  // 4. Concatenate 16 Recency + 8 Type + 4 Phase = 28-dim State
  std::vector<float> state = recency_norm;
  state.insert(state.end(), type_onehot.begin(), type_onehot.end());
  state.insert(state.end(), phase_onehot.begin(), phase_onehot.end());

  // 5. DQN Forward Pass
  std::vector<float> q_values = state;
  for (std::size_t i = 0; i < dqn_layers.size(); ++i) {
    bool is_last = (i == dqn_layers.size() - 1);
    q_values = forward_layer(q_values, dqn_layers[i], !is_last);
  }

  // 6. Argmax Q-value to get predicted RECENCY RANK (0..15, 15=LRU)
  auto max_it = std::max_element(q_values.begin(), q_values.end());
  long target_rank = std::distance(q_values.begin(), max_it);

  // Map predicted recency rank to the physical way currently holding that rank
  long best_way = 0;
  long min_diff = 100;
  for (long w = 0; w < NUM_WAY; ++w) {
    long r = std::lround(recency_norm[w] * (NUM_WAY - 1));
    long diff = std::abs(r - target_rank);
    if (diff < min_diff) {
      min_diff = diff;
      best_way = w;
    }
  }

  static int debug_q = 0;
  if (debug_q < 10) {
    std::cout << "[DQN PREDICTION " << debug_q << "] Target Rank: " << target_rank << " -> Evicting Physical Way: " << best_way << std::endl;
    debug_q++;
  }

  return (best_way >= 0 && best_way < NUM_WAY) ? best_way : 0;
}

long phase_aware_rl::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                                 champsim::address full_addr, access_type type)
{
  uint64_t current_cpu_cycle = 0;
  if (intern_ && intern_->clock_period.count() > 0) {
    current_cpu_cycle = static_cast<uint64_t>(intern_->current_time.time_since_epoch() / intern_->clock_period);
  }

  static uint64_t dqn_invocations = 0;
  if (weights_loaded) {
    long victim = predict_victim_way(set, type, current_cpu_cycle);
    if (dqn_invocations < 5) {
      std::cout << "[DQN PREDICTION " << dqn_invocations << "] Set: " << set << " -> Evicting Way: " << victim << std::endl;
      dqn_invocations++;
    }
    return victim;
  }

  // Fallback to LRU
  auto begin = std::next(std::begin(last_used_cycles), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);
  auto victim = std::min_element(begin, end);
  return std::distance(begin, victim);
}

void phase_aware_rl::replacement_cache_fill(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
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

void phase_aware_rl::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
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
}

void phase_aware_rl::replacement_final_stats()
{
  std::cout << "[phase_aware_rl] Final Stats: Accesses=" << total_accesses << ", Misses=" << total_misses << "\n";
}
