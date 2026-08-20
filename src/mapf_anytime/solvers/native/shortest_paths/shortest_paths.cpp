#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct Grid {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> passable;
};

struct AgentPair {
    int start = -1;
    int goal = -1;
};

struct ProcessedGrid {
    Grid grid;
    int offset_x = 0;
    int offset_y = 0;
    int free_cells = 0;
};

bool is_passable(const char tile) {
    return tile == '.' || tile == 'G' || tile == 'S' || tile == 'W';
}

Grid load_map(const std::string& map_path) {
    std::ifstream input(map_path);
    if (!input) {
        throw std::runtime_error("Could not open map: " + map_path);
    }

    Grid grid;
    std::string key;
    std::string line;
    bool found_map = false;
    while (std::getline(input, line)) {
        if (line == "map") {
            found_map = true;
            break;
        }
        const auto separator = line.find(' ');
        if (separator == std::string::npos) {
            continue;
        }
        key = line.substr(0, separator);
        const auto value = line.substr(separator + 1);
        if (key == "width") {
            grid.width = std::stoi(value);
        } else if (key == "height") {
            grid.height = std::stoi(value);
        }
    }

    if (!found_map || grid.width <= 0 || grid.height <= 0) {
        throw std::runtime_error("Malformed MovingAI map header: " + map_path);
    }

    grid.passable.assign(
        static_cast<std::size_t>(grid.width) * grid.height,
        0
    );
    for (int y = 0; y < grid.height; ++y) {
        if (!std::getline(input, line)) {
            throw std::runtime_error("Map has fewer rows than declared: " + map_path);
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (static_cast<int>(line.size()) != grid.width) {
            throw std::runtime_error("Map row width does not match header: " + map_path);
        }
        for (int x = 0; x < grid.width; ++x) {
            grid.passable[static_cast<std::size_t>(y) * grid.width + x] =
                static_cast<std::uint8_t>(is_passable(line[x]));
        }
    }
    return grid;
}

std::vector<AgentPair> load_scenario(
    const std::string& scenario_path,
    const Grid& grid,
    const std::size_t agent_count
) {
    std::ifstream input(scenario_path);
    if (!input) {
        throw std::runtime_error("Could not open scenario: " + scenario_path);
    }

    std::string version;
    std::getline(input, version);
    if (version.rfind("version", 0) != 0) {
        throw std::runtime_error("Malformed MovingAI scenario header: " + scenario_path);
    }

    std::vector<AgentPair> pairs;
    pairs.reserve(agent_count);
    std::string map_name;
    int bucket = 0;
    int scenario_width = 0;
    int scenario_height = 0;
    int start_x = 0;
    int start_y = 0;
    int goal_x = 0;
    int goal_y = 0;
    double stored_distance = 0;

    while (
        pairs.size() < agent_count
        && input >> bucket >> map_name >> scenario_width >> scenario_height
                 >> start_x >> start_y >> goal_x >> goal_y >> stored_distance
    ) {
        if (scenario_width != grid.width || scenario_height != grid.height) {
            throw std::runtime_error(
                "Scenario dimensions do not match map dimensions: " + scenario_path
            );
        }
        if (
            start_x < 0 || start_x >= grid.width
            || start_y < 0 || start_y >= grid.height
            || goal_x < 0 || goal_x >= grid.width
            || goal_y < 0 || goal_y >= grid.height
        ) {
            throw std::runtime_error("Scenario coordinate is outside the map");
        }
        const int start = start_y * grid.width + start_x;
        const int goal = goal_y * grid.width + goal_x;
        if (!grid.passable[start] || !grid.passable[goal]) {
            throw std::runtime_error("Scenario start or goal is blocked");
        }
        pairs.push_back({start, goal});
    }

    if (pairs.size() != agent_count) {
        throw std::runtime_error(
            "Scenario contains fewer rows than the requested agent count"
        );
    }
    return pairs;
}

ProcessedGrid preprocess_grid(const Grid& source) {
    const int source_size = source.width * source.height;
    std::vector<std::uint8_t> visited(source_size, 0);
    std::vector<int> queue(source_size);
    std::vector<int> largest_component;
    std::vector<int> component;

    for (int start = 0; start < source_size; ++start) {
        if (!source.passable[start] || visited[start]) {
            continue;
        }
        component.clear();
        std::size_t begin = 0;
        std::size_t end = 1;
        queue[0] = start;
        visited[start] = 1;
        while (begin < end) {
            const int current = queue[begin++];
            component.push_back(current);
            const int x = current % source.width;
            const int y = current / source.width;
            const int candidates[4] = {
                x > 0 ? current - 1 : -1,
                x + 1 < source.width ? current + 1 : -1,
                y > 0 ? current - source.width : -1,
                y + 1 < source.height ? current + source.width : -1,
            };
            for (const int next : candidates) {
                if (next >= 0 && source.passable[next] && !visited[next]) {
                    visited[next] = 1;
                    queue[end++] = next;
                }
            }
        }
        if (component.size() > largest_component.size()) {
            largest_component = component;
        }
    }

    if (largest_component.empty()) {
        throw std::runtime_error("Map contains no connected free-space component");
    }

    int min_x = source.width;
    int max_x = 0;
    int min_y = source.height;
    int max_y = 0;
    for (const int location : largest_component) {
        const int x = location % source.width;
        const int y = location / source.width;
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    ProcessedGrid processed;
    processed.offset_x = min_x;
    processed.offset_y = min_y;
    processed.free_cells = static_cast<int>(largest_component.size());
    processed.grid.width = max_x - min_x + 1;
    processed.grid.height = max_y - min_y + 1;
    processed.grid.passable.assign(
        static_cast<std::size_t>(processed.grid.width) * processed.grid.height,
        0
    );
    for (const int location : largest_component) {
        const int x = location % source.width - min_x;
        const int y = location / source.width - min_y;
        processed.grid.passable[
            static_cast<std::size_t>(y) * processed.grid.width + x
        ] = 1;
    }
    return processed;
}

std::vector<AgentPair> transform_pairs(
    const std::vector<AgentPair>& source_pairs,
    const Grid& source,
    const ProcessedGrid& processed
) {
    std::vector<AgentPair> result;
    result.reserve(source_pairs.size());
    for (const auto& pair : source_pairs) {
        const int start_x = pair.start % source.width - processed.offset_x;
        const int start_y = pair.start / source.width - processed.offset_y;
        const int goal_x = pair.goal % source.width - processed.offset_x;
        const int goal_y = pair.goal / source.width - processed.offset_y;
        if (
            start_x < 0 || start_x >= processed.grid.width
            || start_y < 0 || start_y >= processed.grid.height
            || goal_x < 0 || goal_x >= processed.grid.width
            || goal_y < 0 || goal_y >= processed.grid.height
        ) {
            throw std::runtime_error(
                "Scenario start or goal lies outside the processed map"
            );
        }
        const int start = start_y * processed.grid.width + start_x;
        const int goal = goal_y * processed.grid.width + goal_x;
        if (
            !processed.grid.passable[start]
            || !processed.grid.passable[goal]
        ) {
            throw std::runtime_error(
                "Scenario start or goal is outside the largest free component"
            );
        }
        result.push_back({start, goal});
    }
    return result;
}

class BidirectionalBfs {
public:
    explicit BidirectionalBfs(const std::size_t map_size)
        : forward_seen_(map_size, 0),
          backward_seen_(map_size, 0),
          forward_distance_(map_size, 0),
          backward_distance_(map_size, 0),
          forward_queue_(map_size),
          backward_queue_(map_size) {}

    int distance(const Grid& grid, const int start, const int goal) {
        if (start == goal) {
            return 0;
        }
        next_generation();

        std::size_t forward_begin = 0;
        std::size_t forward_end = 1;
        std::size_t backward_begin = 0;
        std::size_t backward_end = 1;
        forward_queue_[0] = start;
        backward_queue_[0] = goal;
        forward_seen_[start] = generation_;
        backward_seen_[goal] = generation_;
        forward_distance_[start] = 0;
        backward_distance_[goal] = 0;
        int best_distance = std::numeric_limits<int>::max();

        while (forward_begin < forward_end && backward_begin < backward_end) {
            const int forward_min_distance =
                forward_distance_[forward_queue_[forward_begin]];
            const int backward_min_distance =
                backward_distance_[backward_queue_[backward_begin]];
            if (
                best_distance != std::numeric_limits<int>::max()
                && forward_min_distance + backward_min_distance >= best_distance
            ) {
                return best_distance;
            }

            const auto forward_layer_end = forward_end;
            const auto backward_layer_end = backward_end;
            const auto forward_layer_size = forward_layer_end - forward_begin;
            const auto backward_layer_size = backward_layer_end - backward_begin;

            if (forward_layer_size <= backward_layer_size) {
                const int result = expand_layer(
                    grid,
                    forward_begin,
                    forward_end,
                    forward_layer_end,
                    forward_queue_,
                    forward_seen_,
                    forward_distance_,
                    backward_seen_,
                    backward_distance_
                );
                if (result >= 0) {
                    best_distance = std::min(best_distance, result);
                }
            } else {
                const int result = expand_layer(
                    grid,
                    backward_begin,
                    backward_end,
                    backward_layer_end,
                    backward_queue_,
                    backward_seen_,
                    backward_distance_,
                    forward_seen_,
                    forward_distance_
                );
                if (result >= 0) {
                    best_distance = std::min(best_distance, result);
                }
            }
        }
        return best_distance == std::numeric_limits<int>::max()
            ? -1
            : best_distance;
    }

private:
    std::vector<std::uint32_t> forward_seen_;
    std::vector<std::uint32_t> backward_seen_;
    std::vector<int> forward_distance_;
    std::vector<int> backward_distance_;
    std::vector<int> forward_queue_;
    std::vector<int> backward_queue_;
    std::uint32_t generation_ = 0;

    void next_generation() {
        ++generation_;
        if (generation_ == 0) {
            std::fill(forward_seen_.begin(), forward_seen_.end(), 0);
            std::fill(backward_seen_.begin(), backward_seen_.end(), 0);
            generation_ = 1;
        }
    }

    int expand_layer(
        const Grid& grid,
        std::size_t& queue_begin,
        std::size_t& queue_end,
        const std::size_t layer_end,
        std::vector<int>& queue,
        std::vector<std::uint32_t>& own_seen,
        std::vector<int>& own_distance,
        const std::vector<std::uint32_t>& other_seen,
        const std::vector<int>& other_distance
    ) {
        int best_distance = std::numeric_limits<int>::max();
        while (queue_begin < layer_end) {
            const int current = queue[queue_begin++];
            const int next_distance = own_distance[current] + 1;
            const int x = current % grid.width;
            const int y = current / grid.width;

            const int candidates[4] = {
                x > 0 ? current - 1 : -1,
                x + 1 < grid.width ? current + 1 : -1,
                y > 0 ? current - grid.width : -1,
                y + 1 < grid.height ? current + grid.width : -1,
            };
            for (const int next : candidates) {
                if (
                    next < 0
                    || !grid.passable[next]
                    || own_seen[next] == generation_
                ) {
                    continue;
                }
                own_seen[next] = generation_;
                own_distance[next] = next_distance;
                if (other_seen[next] == generation_) {
                    best_distance = std::min(
                        best_distance,
                        next_distance + other_distance[next]
                    );
                }
                queue[queue_end++] = next;
            }
        }
        return best_distance == std::numeric_limits<int>::max()
            ? -1
            : best_distance;
    }
};

std::vector<int> shortest_path_distances(
    const std::string& map_path,
    const std::string& scenario_path,
    const std::size_t agent_count,
    int num_threads
) {
    if (agent_count == 0) {
        return {};
    }
    const Grid grid = load_map(map_path);
    const auto pairs = load_scenario(scenario_path, grid, agent_count);

    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    num_threads = std::max(1, num_threads);
    num_threads = std::min<int>(num_threads, static_cast<int>(agent_count));

    std::vector<int> distances(agent_count, -1);
    std::atomic<std::size_t> next_agent{0};
    std::atomic<bool> unreachable{false};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    {
        py::gil_scoped_release release;
        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            workers.emplace_back([&]() {
                BidirectionalBfs search(grid.passable.size());
                while (true) {
                    const std::size_t agent =
                        next_agent.fetch_add(1, std::memory_order_relaxed);
                    if (agent >= agent_count) {
                        break;
                    }
                    const int distance = search.distance(
                        grid,
                        pairs[agent].start,
                        pairs[agent].goal
                    );
                    distances[agent] = distance;
                    if (distance < 0) {
                        unreachable.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }

    if (unreachable.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
            "At least one start-goal pair is unreachable using 4-neighbor movement"
        );
    }
    return distances;
}

class FeatureWorkspace {
public:
    explicit FeatureWorkspace(const std::size_t map_size)
        : goal_distance_(map_size, -1),
          start_distance_(map_size, -1),
          queue_(map_size),
          marked_(map_size, 0) {}

    int compute(
        const Grid& grid,
        const AgentPair& pair,
        const bool exact_cells
    ) {
        bfs(grid, pair.goal, goal_distance_);
        const int shortest = goal_distance_[pair.start];
        if (shortest < 0) {
            return -1;
        }

        if (exact_cells) {
            bfs(grid, pair.start, start_distance_);
            for (std::size_t cell = 0; cell < grid.passable.size(); ++cell) {
                if (
                    grid.passable[cell]
                    && start_distance_[cell] >= 0
                    && goal_distance_[cell] >= 0
                    && start_distance_[cell] + goal_distance_[cell] == shortest
                ) {
                    marked_[cell] = 1;
                }
            }
        } else {
            int current = pair.start;
            marked_[current] = 1;
            while (current != pair.goal) {
                const int current_distance = goal_distance_[current];
                const int x = current % grid.width;
                const int y = current / grid.width;
                const int candidates[4] = {
                    x > 0 ? current - 1 : -1,
                    x + 1 < grid.width ? current + 1 : -1,
                    y > 0 ? current - grid.width : -1,
                    y + 1 < grid.height ? current + grid.width : -1,
                };
                int next_step = -1;
                for (const int next : candidates) {
                    if (
                        next >= 0
                        && grid.passable[next]
                        && goal_distance_[next] == current_distance - 1
                    ) {
                        next_step = next;
                        break;
                    }
                }
                if (next_step < 0) {
                    throw std::runtime_error(
                        "Could not reconstruct a shortest path"
                    );
                }
                current = next_step;
                marked_[current] = 1;
            }
        }
        return shortest;
    }

    const std::vector<std::uint8_t>& marked() const {
        return marked_;
    }

private:
    std::vector<int> goal_distance_;
    std::vector<int> start_distance_;
    std::vector<int> queue_;
    std::vector<std::uint8_t> marked_;

    void bfs(const Grid& grid, const int source, std::vector<int>& distance) {
        std::fill(distance.begin(), distance.end(), -1);
        std::size_t begin = 0;
        std::size_t end = 1;
        queue_[0] = source;
        distance[source] = 0;
        while (begin < end) {
            const int current = queue_[begin++];
            const int next_distance = distance[current] + 1;
            const int x = current % grid.width;
            const int y = current / grid.width;
            const int candidates[4] = {
                x > 0 ? current - 1 : -1,
                x + 1 < grid.width ? current + 1 : -1,
                y > 0 ? current - grid.width : -1,
                y + 1 < grid.height ? current + grid.width : -1,
            };
            for (const int next : candidates) {
                if (
                    next >= 0
                    && grid.passable[next]
                    && distance[next] < 0
                ) {
                    distance[next] = next_distance;
                    queue_[end++] = next;
                }
            }
        }
    }
};

class ShortestPathWorkspace {
public:
    explicit ShortestPathWorkspace(const std::size_t map_size)
        : goal_distance_(map_size, -1),
          queue_(map_size) {}

    std::vector<int> path(const Grid& grid, const AgentPair& pair) {
        bfs(grid, pair.goal);
        const int shortest = goal_distance_[pair.start];
        if (shortest < 0) {
            return {};
        }

        std::vector<int> result;
        result.reserve(static_cast<std::size_t>(shortest) + 1);
        int current = pair.start;
        result.push_back(current);
        while (current != pair.goal) {
            const int current_distance = goal_distance_[current];
            const int x = current % grid.width;
            const int y = current / grid.width;
            const int candidates[4] = {
                x > 0 ? current - 1 : -1,
                x + 1 < grid.width ? current + 1 : -1,
                y > 0 ? current - grid.width : -1,
                y + 1 < grid.height ? current + grid.width : -1,
            };

            int next_step = -1;
            for (const int next : candidates) {
                if (
                    next >= 0
                    && grid.passable[next]
                    && goal_distance_[next] == current_distance - 1
                ) {
                    next_step = next;
                    break;
                }
            }
            if (next_step < 0) {
                throw std::runtime_error("Could not reconstruct a shortest path");
            }
            current = next_step;
            result.push_back(current);
        }
        return result;
    }

private:
    std::vector<int> goal_distance_;
    std::vector<int> queue_;

    void bfs(const Grid& grid, const int source) {
        std::fill(goal_distance_.begin(), goal_distance_.end(), -1);
        std::size_t begin = 0;
        std::size_t end = 1;
        queue_[0] = source;
        goal_distance_[source] = 0;
        while (begin < end) {
            const int current = queue_[begin++];
            const int next_distance = goal_distance_[current] + 1;
            const int x = current % grid.width;
            const int y = current / grid.width;
            const int candidates[4] = {
                x > 0 ? current - 1 : -1,
                x + 1 < grid.width ? current + 1 : -1,
                y > 0 ? current - grid.width : -1,
                y + 1 < grid.height ? current + grid.width : -1,
            };
            for (const int next : candidates) {
                if (
                    next >= 0
                    && grid.passable[next]
                    && goal_distance_[next] < 0
                ) {
                    goal_distance_[next] = next_distance;
                    queue_[end++] = next;
                }
            }
        }
    }
};

long long pair_count(const long long count) {
    return count * (count - 1) / 2;
}

long long edge_key(const int from, const int to) {
    return (static_cast<long long>(from) << 32)
        | static_cast<unsigned int>(to);
}

py::dict shortest_path_collision_counts(
    const std::string& map_path,
    const std::string& scenario_path,
    const std::size_t agent_count,
    int num_threads
) {
    const Grid source_grid = load_map(map_path);
    const auto source_pairs =
        load_scenario(scenario_path, source_grid, agent_count);
    const ProcessedGrid processed = preprocess_grid(source_grid);
    const auto pairs = transform_pairs(source_pairs, source_grid, processed);

    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    num_threads = std::max(1, num_threads);
    num_threads = std::min<int>(
        num_threads,
        std::max<int>(1, static_cast<int>(agent_count))
    );

    std::vector<std::vector<int>> paths(agent_count);
    std::atomic<std::size_t> next_agent{0};
    std::atomic<bool> unreachable{false};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    {
        py::gil_scoped_release release;
        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            workers.emplace_back([&]() {
                ShortestPathWorkspace workspace(processed.grid.passable.size());
                while (true) {
                    const std::size_t agent =
                        next_agent.fetch_add(1, std::memory_order_relaxed);
                    if (agent >= agent_count) {
                        break;
                    }
                    paths[agent] = workspace.path(processed.grid, pairs[agent]);
                    if (paths[agent].empty()) {
                        unreachable.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }

    if (unreachable.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
            "At least one start-goal pair is unreachable using 4-neighbor movement"
        );
    }

    std::size_t horizon = 0;
    for (const auto& path : paths) {
        horizon = std::max(horizon, path.size());
    }

    long long vertex_collisions = 0;
    long long edge_collisions = 0;
    std::unordered_map<int, int> vertex_counts;
    std::unordered_map<long long, int> edge_counts;
    vertex_counts.reserve(agent_count * 2 + 1);
    edge_counts.reserve(agent_count * 2 + 1);

    for (std::size_t timestep = 0; timestep < horizon; ++timestep) {
        vertex_counts.clear();
        for (const auto& path : paths) {
            const int location =
                timestep < path.size() ? path[timestep] : path.back();
            ++vertex_counts[location];
        }
        for (const auto& [location, count] : vertex_counts) {
            vertex_collisions += pair_count(count);
        }

        if (timestep == 0) {
            continue;
        }

        edge_counts.clear();
        for (const auto& path : paths) {
            const int previous =
                timestep - 1 < path.size() ? path[timestep - 1] : path.back();
            const int current =
                timestep < path.size() ? path[timestep] : path.back();
            if (previous != current) {
                ++edge_counts[edge_key(previous, current)];
            }
        }
        for (const auto& [key, count] : edge_counts) {
            const int from = static_cast<int>(key >> 32);
            const int to = static_cast<int>(key & 0xffffffff);
            if (from < to) {
                const auto opposite = edge_counts.find(edge_key(to, from));
                if (opposite != edge_counts.end()) {
                    edge_collisions +=
                        static_cast<long long>(count) * opposite->second;
                }
            }
        }
    }

    py::dict result;
    result["sp_collision_count"] = vertex_collisions + edge_collisions;
    result["sp_vertex_collision_count"] = vertex_collisions;
    result["sp_edge_collision_count"] = edge_collisions;
    result["sp_horizon"] = horizon > 0 ? static_cast<int>(horizon - 1) : 0;
    return result;
}

py::dict instance_features(
    const std::string& map_path,
    const std::string& scenario_path,
    const std::size_t agent_count,
    const std::string& cells_mode,
    int num_threads
) {
    if (cells_mode != "single-path" && cells_mode != "exact") {
        throw std::invalid_argument(
            "cells_mode must be 'single-path' or 'exact'"
        );
    }
    const Grid source_grid = load_map(map_path);
    const auto source_pairs =
        load_scenario(scenario_path, source_grid, agent_count);
    const ProcessedGrid processed = preprocess_grid(source_grid);
    const auto pairs = transform_pairs(source_pairs, source_grid, processed);

    if (num_threads <= 0) {
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
    }
    num_threads = std::max(1, num_threads);
    num_threads = std::min<int>(
        num_threads,
        std::max<int>(1, static_cast<int>(agent_count))
    );

    std::vector<int> distances(agent_count, -1);
    std::vector<std::vector<std::uint8_t>> thread_marks(
        num_threads,
        std::vector<std::uint8_t>(processed.grid.passable.size(), 0)
    );
    std::atomic<std::size_t> next_agent{0};
    std::atomic<bool> unreachable{false};
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    const bool exact_cells = cells_mode == "exact";

    {
        py::gil_scoped_release release;
        for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            workers.emplace_back([&, thread_id]() {
                FeatureWorkspace workspace(processed.grid.passable.size());
                while (true) {
                    const std::size_t agent =
                        next_agent.fetch_add(1, std::memory_order_relaxed);
                    if (agent >= agent_count) {
                        break;
                    }
                    const int distance = workspace.compute(
                        processed.grid,
                        pairs[agent],
                        exact_cells
                    );
                    distances[agent] = distance;
                    if (distance < 0) {
                        unreachable.store(true, std::memory_order_relaxed);
                    }
                }
                thread_marks[thread_id] = workspace.marked();
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }

    if (unreachable.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
            "At least one start-goal pair is unreachable using 4-neighbor movement"
        );
    }

    std::size_t cells_at_sp = 0;
    for (std::size_t cell = 0; cell < processed.grid.passable.size(); ++cell) {
        bool marked = false;
        for (const auto& marks : thread_marks) {
            if (marks[cell]) {
                marked = true;
                break;
            }
        }
        cells_at_sp += static_cast<std::size_t>(marked);
    }

    long long lower_bound_soc = 0;
    int minimum = 0;
    int maximum = 0;
    if (!distances.empty()) {
        minimum = *std::min_element(distances.begin(), distances.end());
        maximum = *std::max_element(distances.begin(), distances.end());
        for (const int distance : distances) {
            lower_bound_soc += distance;
        }
    }
    const int total_cells =
        processed.grid.width * processed.grid.height;
    const int obstacles = total_cells - processed.free_cells;

    py::dict result;
    result["num_agents"] = agent_count;
    result["num_obstacles"] = obstacles;
    result["agent_density"] = processed.free_cells > 0
        ? static_cast<double>(agent_count) / processed.free_cells
        : 0.0;
    result["obstacle_density"] = total_cells > 0
        ? static_cast<double>(obstacles) / total_cells
        : 0.0;
    result["avg_shortest_path_distance"] = agent_count > 0
        ? static_cast<double>(lower_bound_soc) / agent_count
        : 0.0;
    result["min_shortest_path_distance"] = minimum;
    result["max_shortest_path_distance"] = maximum;
    result["cells_at_sp_ratio"] = processed.free_cells > 0
        ? static_cast<double>(cells_at_sp) / processed.free_cells
        : 0.0;
    result["num_total_cells"] = total_cells;
    result["num_free_cells"] = processed.free_cells;
    result["lower_bound_soc"] = lower_bound_soc;
    return result;
}

py::dict analyze_instance(
    const std::string& map_path,
    const std::string& scenario_path,
    const std::size_t agent_count,
    const std::string& cells_mode,
    int num_threads
) {
    py::dict analysis = instance_features(
        map_path,
        scenario_path,
        agent_count,
        cells_mode,
        num_threads
    );
    const py::dict collisions = shortest_path_collision_counts(
        map_path,
        scenario_path,
        agent_count,
        num_threads
    );
    analysis["sp_collision_count"] = collisions["sp_collision_count"];
    analysis["sp_vertex_collision_count"] =
        collisions["sp_vertex_collision_count"];
    analysis["sp_edge_collision_count"] = collisions["sp_edge_collision_count"];
    analysis["shortest_path_distances"] = shortest_path_distances(
        map_path,
        scenario_path,
        agent_count,
        num_threads
    );
    return analysis;
}

}  // namespace

PYBIND11_MODULE(mapf_shortest_paths, module) {
    module.doc() = "Static features for MovingAI MAPF instances.";
    module.def(
        "analyze_instance",
        &analyze_instance,
        py::arg("map_path"),
        py::arg("scenario_path"),
        py::arg("agent_count"),
        py::arg("cells_mode") = "single-path",
        py::arg("num_threads") = 1,
        "Return static features, shortest-path collisions, and per-agent distances."
    );
}
