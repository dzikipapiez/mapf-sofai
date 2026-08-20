#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "planner.hpp"
#include "post_processing.hpp"
#include "sipp.hpp"
#include "utils.hpp"

Solution solve(const Instance &ins, const int verbose = 0,
               Deadline *deadline = nullptr, int seed = 0,
               std::vector<std::pair<int, int>> *trace = nullptr,
               Solution *initial_solution = nullptr,
               double post_solution_time_limit_ms = -1);
