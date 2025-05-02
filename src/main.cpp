#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>
#include <queue>
#include <memory>

#include <absl/log/log.h>
#include <absl/log/initialize.h>
#include <absl/log/log_entry.h>
#include <absl/log/globals.h>
#include <absl/container/inlined_vector.h>

using Address = uint64_t;
using GHR = uint64_t;

enum class BranchType {
  Conditional,
  Unconditional,
  Call,
  Return,
  Indirect,
};

enum class RequirementType {
  Equal,
  AtLeast,
};

struct Requirement {
  bool outcome;
  std::optional<GHR> ghr;
  uint32_t target_count;
  RequirementType type{RequirementType::Equal};

  bool operator==(const Requirement &other) const = default;
};

struct Branch {
  Address address;
  BranchType type;
  std::optional<Address> target1;
  std::optional<Address> target2;
  std::vector<Requirement> requirements;
  std::vector<std::pair<size_t, size_t>> requirement_indices;

  bool is_valid() const {
    switch (type) {
      case BranchType::Conditional:
        return target1.has_value() && target2.has_value();
      case BranchType::Unconditional:
      case BranchType::Call:
        return target1.has_value() && !target2.has_value();
      case BranchType::Return:
      case BranchType::Indirect:
        return !target2.has_value();
    }
    return false;
  }
};

class BranchGraph {
 public:
  using NodeMap = std::map<Address, Branch>;
  using EdgeSet = std::set<std::pair<Address, Address>>;

 private:
  NodeMap nodes_;
  EdgeSet edges_;
  std::vector<Requirement> requirements_;

  BranchGraph() = default;

 public:
  static std::optional<BranchGraph> create(
      const std::vector<Branch> &initial_branches,
      const std::vector<std::pair<Address, Address>> &initial_edges) {
    BranchGraph graph;
    size_t current_req_id = 0;

    for (const auto &branch : initial_branches) {
      if (!branch.is_valid()) return std::nullopt;

      Branch processed_branch = branch;
      processed_branch.requirement_indices.reserve(
          processed_branch.requirements.size());

      for (size_t i = 0; i < processed_branch.requirements.size(); ++i) {
        if (processed_branch.requirements[i].target_count > 0) {
          processed_branch.requirement_indices.emplace_back(i,
                                                            current_req_id);
          graph.requirements_.push_back(processed_branch.requirements[i]);
          ++current_req_id;
        }
      }

      if (!graph.nodes_
               .emplace(processed_branch.address,
                        std::move(processed_branch))
               .second) {
        return std::nullopt;
      }
    }

    for (const auto &edge : initial_edges) {
      graph.edges_.insert(edge);
    }

    for (const auto &[addr, branch] : graph.nodes_) {
      if (branch.target1) graph.edges_.emplace(addr, *branch.target1);
      if (branch.target2) graph.edges_.emplace(addr, *branch.target2);
    }

    return graph;
  }

  const Branch *get_branch(Address address) const {
    auto it = nodes_.lower_bound(address);
    return it != nodes_.end() ? &it->second : nullptr;
  }

  const Branch *get_branch_exact(Address address) const {
    auto it = nodes_.find(address);
    return it != nodes_.end() ? &it->second : nullptr;
  }

  std::vector<std::pair<Address, Address>> edges_from(
      Address address) const {
    std::vector<std::pair<Address, Address>> result;
    auto it_low = edges_.lower_bound({address, 0});
    auto it_high = edges_.lower_bound({address + 1, 0});
    for (auto it = it_low; it != it_high; ++it) {
      if (it->first == address) result.push_back(*it);
    }
    return result;
  }

  const EdgeSet &all_edges() const { return edges_; }
  size_t total_requirements() const { return requirements_.size(); }
  const Requirement *get_requirement_by_id(size_t req_id) const {
    return req_id < requirements_.size() ? &requirements_[req_id] : nullptr;
  }
  const NodeMap &get_nodes() const { return nodes_; }
};

struct Option {
  int32_t score;
  std::optional<bool> outcome;
  Address next_addr;
  BranchType branch_type;

  bool operator>(const Option &other) const { return score > other.score; }
  bool operator==(const Option &other) const = default;
};

struct PathNode {
  std::shared_ptr<PathNode> parent;
  Option option;

  PathNode(std::shared_ptr<PathNode> p, Option opt)
      : parent(p), option(opt) {}
};

using PathNodePtr = std::shared_ptr<PathNode>;
using RequirementCounts = std::vector<uint32_t>;

struct SearchParams {
  size_t ghr_size;
  size_t max_depth;
  size_t max_queue_size;
  size_t beam_width;
};

int32_t calculate_score(const Branch &branch, std::optional<bool> outcome,
                        GHR current_ghr, size_t ghr_size,
                        const RequirementCounts &current_counts,
                        const BranchGraph &graph) {
  if (!outcome) return 0;

  GHR mask = (1ULL << ghr_size) - 1;
  GHR relevant_ghr = current_ghr & mask;
  int32_t score = 0;

  for (const auto &[idx, req_id] : branch.requirement_indices) {
    const auto &req = branch.requirements[idx];
    const auto *original = graph.get_requirement_by_id(req_id);
    if (!original) continue;

    bool outcome_matches = req.outcome == *outcome;
    bool ghr_matches = !req.ghr || (*req.ghr & mask) == relevant_ghr;

    if (outcome_matches && ghr_matches) {
      if (original->type == RequirementType::Equal) {
        if (current_counts[req_id] < original->target_count) {
          score += 3;
          if (req.ghr && (*req.ghr & mask) == relevant_ghr) score += 2;
        } else if (current_counts[req_id] == original->target_count) {
          score -= 1; // Penalize overshooting exact requirements
        }
      } else if (original->type == RequirementType::AtLeast) {
        if (current_counts[req_id] < original->target_count) {
          score += 3;
          if (req.ghr && (*req.ghr & mask) == relevant_ghr) score += 2;
        } else {
          score += 1; // Still reward maintaining AtLeast requirements
        }
      }
    } else if (original->type == RequirementType::Equal &&
               current_counts[req_id] > 0 && !outcome_matches) {
      score -=
          2; // Stronger penalty for wrong outcomes on Equal requirements
    }
  }
  return score;
}

struct BPState {
  Address current_addr;
  GHR current_ghr;
  RequirementCounts current_counts;
  absl::InlinedVector<Address, 8> ras;
  PathNodePtr path_node;
  size_t depth{0};
  int32_t cumulative_score{0};

  BPState(Address addr, size_t num_requirements)
      : current_addr(addr),
        current_ghr(0),
        current_counts(num_requirements, 0),
        path_node(nullptr) {}

  BPState(const BPState &other) = default;

  std::vector<Option> get_options(const BranchGraph &graph,
                                  size_t ghr_size) const {
    const Branch *branch = graph.get_branch_exact(current_addr);
    if (!branch) return {};

    std::vector<Option> options;

    auto add_option = [&](std::optional<bool> outcome, Address target,
                          BranchType type) {
      options.emplace_back(
          Option{calculate_score(*branch, outcome, current_ghr, ghr_size,
                                 current_counts, graph),
                 outcome, target, type});
    };

    switch (branch->type) {
      case BranchType::Conditional:
        add_option(true, *branch->target2, branch->type);
        add_option(false, *branch->target1, branch->type);
        break;
      case BranchType::Unconditional:
      case BranchType::Call:
        add_option(true, *branch->target1, branch->type);
        break;
      case BranchType::Return:
      case BranchType::Indirect:
        if (branch->type == BranchType::Return && !ras.empty()) {
          if (const Branch *next = graph.get_branch(ras.back())) {
            add_option(std::nullopt, next->address, branch->type);
            break;
          }
        }

        for (const auto &[_, dst] : graph.edges_from(branch->address)) {
          if (graph.get_branch_exact(dst)) {
            add_option(true, dst, branch->type);
          }
        }
        break;
    }

    std::sort(options.begin(), options.end(), std::greater{});
    return options;
  }

  BPState advance(const Option &option, const Branch *branch,
                  const BranchGraph &graph, size_t ghr_size) const {
    BPState next = *this;
    next.path_node = std::make_shared<PathNode>(path_node, option);
    next.depth = depth + 1;
    next.cumulative_score += option.score;

    const GHR mask = (1ULL << ghr_size) - 1;
    if (option.outcome) {
      next.current_ghr =
          ((next.current_ghr << 1) | (*option.outcome ? 1 : 0)) & mask;
      const GHR prev_ghr = current_ghr & mask;

      for (const auto &[idx, req_id] : branch->requirement_indices) {
        const Requirement &req = branch->requirements[idx];
        if (req.outcome != *option.outcome ||
            (req.ghr && (*req.ghr & mask) != prev_ghr))
          continue;

        if (const Requirement *orig = graph.get_requirement_by_id(req_id)) {
          if (next.current_counts[req_id] < orig->target_count) {
            ++next.current_counts[req_id];
          }
        }
      }
    }

    if (option.branch_type == BranchType::Call) {
      next.ras.push_back(branch->address + 1);
    } else if (option.branch_type == BranchType::Return &&
               !next.ras.empty()) {
      next.ras.pop_back();
    }

    next.current_addr = option.next_addr;
    return next;
  }

  bool all_requirements_satisfied(const BranchGraph &graph) const {
    return std::ranges::all_of(
        std::views::iota(0u, graph.total_requirements()), [&](size_t i) {
          const auto *req = graph.get_requirement_by_id(i);
          if (!req) return true;

          switch (req->type) {
            case RequirementType::Equal:
              return current_counts[i] >= req->target_count;
            case RequirementType::AtLeast:
              return current_counts[i] >= req->target_count;
          }
          return false;
        });
  }

  float calculate_progress(const BranchGraph &graph) const {
    if (graph.total_requirements() == 0) return 1.0f;

    float total_progress = 0.0f;
    for (size_t i = 0; i < graph.total_requirements(); ++i) {
      const auto *req = graph.get_requirement_by_id(i);
      if (!req) continue;

      float req_progress = std::min(
          1.0f, static_cast<float>(current_counts[i]) / req->target_count);
      total_progress += req_progress;
    }

    return total_progress / graph.total_requirements();
  }

  size_t state_hash() const {
    size_t seed = 0;

    seed ^= std::hash<Address>{}(current_addr) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<GHR>{}(current_ghr) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);

    for (const auto &count : current_counts) {
      seed ^= std::hash<uint32_t>{}(count) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }

    for (const auto &addr : ras) {
      seed ^= std::hash<Address>{}(addr) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }

    return seed;
  }
};

std::vector<Option> reconstruct_path(PathNodePtr node) {
  std::vector<Option> path;
  while (node) {
    path.push_back(node->option);
    node = node->parent;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

std::optional<std::vector<Option>> find_path_beam_search(
    const BPState &initial_state, const BranchGraph &graph,
    const SearchParams &params) {
  struct StateEntry {
    BPState state;
    float priority;

    bool operator<(const StateEntry &other) const {
      return priority < other.priority;
    }
  };

  std::priority_queue<StateEntry> current_level;
  std::unordered_set<size_t> visited;

  current_level.push({initial_state, 0.0f});

  for (size_t depth = 0; depth < params.max_depth; ++depth) {
    if (current_level.empty()) break;

    std::priority_queue<StateEntry> next_level;
    size_t processed_at_level = 0;

    while (!current_level.empty() &&
           processed_at_level < params.beam_width) {
      StateEntry entry = current_level.top();
      current_level.pop();

      const BPState &state = entry.state;

      size_t state_hash = state.state_hash();
      if (visited.contains(state_hash)) {
        continue;
      }
      visited.insert(state_hash);

      if (state.all_requirements_satisfied(graph)) {
        return reconstruct_path(state.path_node);
      }

      const Branch *branch = graph.get_branch_exact(state.current_addr);
      if (!branch) {
        branch = graph.get_branch(state.current_addr);
        if (!branch) continue;
      }

      auto options = state.get_options(graph, params.ghr_size);

      for (const auto &option : options) {
        BPState next_state =
            state.advance(option, branch, graph, params.ghr_size);

        float progress = next_state.calculate_progress(graph);
        float priority = next_state.cumulative_score + (progress * 10.0f) -
                         (0.1f * next_state.depth);

        if (next_level.size() < params.max_queue_size) {
          next_level.push({std::move(next_state), priority});
        } else if (priority > next_level.top().priority) {
          next_level.pop();
          next_level.push({std::move(next_state), priority});
        }
      }

      processed_at_level++;
    }

    current_level = std::move(next_level);

    if (depth % 100 == 0) {
      LOG(INFO) << "Depth " << depth
                << ", queue size: " << current_level.size()
                << ", visited: " << visited.size();
    }
  }

  return std::nullopt;
}

std::optional<std::vector<Option>> find_path(const Address start_addr,
                                             const BranchGraph &graph,
                                             SearchParams params) {
  BPState initial_state(start_addr, graph.total_requirements());
  return find_path_beam_search(initial_state, graph, params);
}

std::pair<std::vector<Branch>, std::vector<std::pair<Address, Address>>>
create_example_data() {
  std::vector<Branch> branches = {
      {0x1000,
       BranchType::Conditional,
       0x1004,
       0x2000, // target1=NotTaken(false), target2=Taken(true)
       {
           Requirement{true, std::nullopt, 3}, // Taken 3 times (any GHR)
           Requirement{false, std::nullopt, 1} // Not-Taken 1 time (any GHR)
       }},
      {0x1004, BranchType::Unconditional, 0x4000, std::nullopt, {}},
      {0x2000,
       BranchType::Call,
       0x3000,
       std::nullopt,
       {Requirement{true, std::nullopt, 3}}},
      {0x2004, BranchType::Unconditional, 0x4000, std::nullopt, {}},

      {0x3000,
       BranchType::Conditional,
       0x300C,
       0x3004, // target1=NotTaken(false), target2=Taken(true)
       {Requirement{true, std::nullopt, 5},
        Requirement{false, std::nullopt, 4}}},
      {0x3004, BranchType::Unconditional, 0x1000}, // Loop back
      {0x300C, BranchType::Return},

      {0x4000, BranchType::Indirect} // End node / potential indirect target
  };

  std::vector<std::pair<Address, Address>> edges = {{0x4000, 0x1000}};

  return {branches, edges};
}

std::pair<std::vector<Branch>, std::vector<std::pair<Address, Address>>>
create_extreme_cyclic_example() {
  std::vector<Branch> branches = {
    {0x1000, BranchType::Conditional, 0x1004, 0x1010, {
      Requirement{true, std::nullopt, 5}, 
      Requirement{false, std::nullopt, 2} 
    }},
    {0x1004, BranchType::Call, 0x2000, std::nullopt, {
      Requirement{true, std::nullopt, 3}
    }},
    {0x1008, BranchType::Conditional, 0x1000, 0x1010, {
      Requirement{true, std::nullopt, 2},
      Requirement{false, std::nullopt, 1}
    }},

    {0x1010, BranchType::Call, 0x3000, std::nullopt, {
      Requirement{true, std::nullopt, 1}
    }},
    {0x1014, BranchType::Unconditional, 0x1020},

    {0x1020, BranchType::Conditional, 0x1024, 0x1030, {
      Requirement{true, std::nullopt, 2},
      Requirement{false, std::nullopt, 1}
    }},
    {0x1024, BranchType::Call, 0x4000, std::nullopt, {
      Requirement{true, std::nullopt, 1}
    }},
    {0x1028, BranchType::Indirect},

    {0x1030, BranchType::Call, 0x5000, std::nullopt, {
      Requirement{true, std::nullopt, 2}
    }},
    {0x1034, BranchType::Return},

    {0x2000, BranchType::Conditional, 0x2004, 0x2008, {
      Requirement{true, std::nullopt, 2},
      Requirement{false, std::nullopt, 1}
    }},
    {0x2004, BranchType::Unconditional, 0x200C},
    {0x2008, BranchType::Call, 0x6000, std::nullopt, {
      Requirement{true, std::nullopt, 1}
    }},
    {0x200C, BranchType::Return},

    {0x3000, BranchType::Unconditional, 0x3004},
    {0x3004, BranchType::Call, 0x7000, std::nullopt, {
      Requirement{true, std::nullopt, 1}
    }},
    {0x3008, BranchType::Return},

    {0x4000, BranchType::Unconditional, 0x4004},
    {0x4004, BranchType::Return},

    {0x5000, BranchType::Conditional, 0x5004, 0x5010, {
      Requirement{true, std::nullopt, 1},
      Requirement{false, std::nullopt, 2}
    }},
    {0x5004, BranchType::Call, 0x6000, std::nullopt, {
      Requirement{true, std::nullopt, 1}
    }},
    {0x5008, BranchType::Unconditional, 0x500C},
    {0x500C, BranchType::Return},
    {0x5010, BranchType::Return},

    {0x6000, BranchType::Conditional, 0x6004, 0x6010, {
      Requirement{true, std::nullopt, 1},
      Requirement{false, std::nullopt, 1}
    }},
    {0x6004, BranchType::Unconditional, 0x6008},
    {0x6008, BranchType::Return},
    {0x6010, BranchType::Unconditional, 0x1000}, 

    {0x7000, BranchType::Call, 0x4000, std::nullopt, {
      Requirement{true, std::nullopt, 1}
    }},
    {0x7004, BranchType::Return}
  };

  std::vector<std::pair<Address, Address>> edges = {
    {0x1008, 0x1010},     // alt exit from loop
    {0x1014, 0x1020},     // fallthrough from 0x1010 call
    {0x1028, 0x1030},     // indirect jump to next logic
    {0x2008, 0x200C},     // after util call
    {0x3008, 0x1014},     // return from 0x3000 back to main path
    {0x4004, 0x1028},     // return to indirect jump
    {0x5004, 0x5008},     // follow-up jump
    {0x6008, 0x500C},     // return from utility
    {0x6010, 0x1000},     // hard loop back
    {0x7004, 0x3008}      // back into the deeper logic
  };

  return {branches, edges};
}

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  auto [branches, edges] = create_extreme_cyclic_example();
  auto graph_opt = BranchGraph::create(branches, edges);

  if (!graph_opt) {
    std::cerr << "Failed to create branch graph.\n";
    return 1;
  }

  auto path_opt = find_path(0x1000, *graph_opt,
                            SearchParams{.ghr_size = 4,
                                         .max_depth = 1000,
                                         .max_queue_size = 50000,
                                         .beam_width = 1000});

  if (path_opt && !path_opt->empty()) {
    const auto &path = *path_opt;
    std::cout << "Path found (" << path.size() << " steps):\n";
    for (const auto &step : path) {
      std::cout << "  0x" << std::hex << step.next_addr << " "
                << (step.outcome ? (*step.outcome ? "Taken" : "NotTaken")
                                 : "Return")
                << " (score: " << std::dec << step.score << ")\n";
    }
  } else {
    std::cout << "No path found satisfying all requirements.\n";
  }
  return 0;
}
