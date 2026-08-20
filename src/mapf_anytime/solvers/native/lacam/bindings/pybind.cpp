#include <lacam.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

using RawSolution = std::vector<std::vector<std::vector<int>>>;
using RawTrace = std::vector<std::pair<int, int>>;

RawSolution convert_solution(const Solution& solution)
{
    const auto agent_count = solution.front().size();
    RawSolution paths(agent_count);
    for (const auto& timestep : solution)
        for (std::size_t agent = 0; agent < agent_count; ++agent)
            paths[agent].push_back({timestep[agent]->x, timestep[agent]->y});
    return paths;
}

std::tuple<RawSolution, RawTrace, RawSolution> solve_with_trace_py(
    const std::vector<std::vector<int>>& map,
    const std::vector<std::pair<int, int>>& starts,
    const std::vector<std::pair<int, int>>& goals,
    const double timeout,
    const bool anytime, const int seed, const double post_solution_seconds,
    const int pibt_num, const bool parallel_pibt)
{
    if (pibt_num < 1)
        throw std::invalid_argument("pibt_num must be positive");
    Instance instance(map, starts, goals);
    if (!instance.is_valid(1))
        return {{}, {}, {}};

    Planner::FLG_SWAP = true;
    Planner::FLG_STAR = anytime;
    Planner::FLG_MULTI_THREAD = parallel_pibt && pibt_num > 1;
    Planner::PIBT_NUM = pibt_num;
    Planner::FLG_REFINER = anytime;
    Planner::REFINER_NUM = 4;
    Planner::FLG_SCATTER = true;
    Planner::SCATTER_MARGIN = 10;
    Planner::RANDOM_INSERT_PROB1 = 0.001;
    Planner::RANDOM_INSERT_PROB2 = 0.01;
    Planner::FLG_RANDOM_INSERT_INIT_NODE = false;
    Planner::RECURSIVE_RATE = 0.2;
    Planner::RECURSIVE_TIME_LIMIT = 1000;
    Planner::CHECKPOINTS_DURATION = 5000;

    auto deadline = Deadline(timeout * 1000);
    RawTrace trace;
    Solution initial_solution;
    const auto solution =
        ::solve(instance, 0, &deadline, seed, &trace, &initial_solution,
                post_solution_seconds * 1000);
    if (solution.empty() || !is_feasible_solution(instance, solution, 0))
        return {{}, {}, {}};
    return {
        convert_solution(solution),
        trace,
        convert_solution(initial_solution),
    };
}

RawSolution solve_py(
    const std::vector<std::vector<int>>& map,
    const std::vector<std::pair<int, int>>& starts,
    const std::vector<std::pair<int, int>>& goals,
    const double timeout,
    const bool anytime, const int seed, const double post_solution_seconds,
    const int pibt_num, const bool parallel_pibt)
{
    return std::get<0>(
        solve_with_trace_py(map, starts, goals, timeout, anytime, seed,
                            post_solution_seconds, pibt_num, parallel_pibt));
}

PYBIND11_MODULE(lacam, module)
{
    module.def(
        "solve",
        &solve_py,
        py::arg("map"),
        py::arg("starts"),
        py::arg("goals"),
        py::arg("timeout"),
        py::arg("anytime") = true, py::arg("seed") = 0,
        py::arg("post_solution_seconds") = -1,
        py::arg("pibt_num") = 1,
        py::arg("parallel_pibt") = true);
    module.def(
        "solve_with_trace",
        &solve_with_trace_py,
        py::arg("map"),
        py::arg("starts"),
        py::arg("goals"),
        py::arg("timeout"),
        py::arg("anytime") = true, py::arg("seed") = 0,
        py::arg("post_solution_seconds") = -1,
        py::arg("pibt_num") = 1,
        py::arg("parallel_pibt") = true);
}
