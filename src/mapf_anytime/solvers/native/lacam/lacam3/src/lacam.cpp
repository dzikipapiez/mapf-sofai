#include "../include/lacam.hpp"

Solution solve(const Instance &ins, int verbose, Deadline *deadline,
               int seed, std::vector<std::pair<int, int>> *trace,
               Solution *initial_solution, double post_solution_time_limit_ms)
{
  info(1, verbose, deadline, "pre-processing");
  auto planner =
      Planner(&ins, verbose, deadline, seed, 0, nullptr,
              post_solution_time_limit_ms);
  auto solution = planner.solve();
  if (trace != nullptr) *trace = planner.trace;
  if (initial_solution != nullptr) *initial_solution = planner.initial_solution;
  return solution;
}
