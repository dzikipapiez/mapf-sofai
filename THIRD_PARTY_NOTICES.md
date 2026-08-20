# Third-party software

MAPF Anytime is an independent codebase and contains no software copied from
SOFAI. The SOFAI licence therefore does not apply to this repository.

The native solver sources below are redistributed third-party software. Their
original licences remain in force and are included beside the corresponding
source trees.

## MAPF-LNS

- Upstream: [Jiaoyang-Li/MAPF-LNS](https://github.com/Jiaoyang-Li/MAPF-LNS)
- Compared revision: `95785de66f8fbdb91fed871e1a56c8c563f39b1f`
- Copyright: © 2021 The University of Southern California
- Licence: USC Research License; educational, research and non-profit use is
  permitted under its terms. Commercial use requires separate permission.
- Bundled licence:
  [`src/mapf_anytime/solvers/native/mapf_lns/license.txt`](src/mapf_anytime/solvers/native/mapf_lns/license.txt)

The bundled version retains MAPF-LNS's search and neighbourhood-selection
implementation. It differs from the upstream revision in the following
integration points:

1. It accepts a complete external initial solution, validates its paths, and
   installs it as the initial LNS incumbent. This is how a LaCAM solution is
   passed into MAPF-LNS.
2. It exposes an explicit random seed rather than seeding from wall-clock
   time.
3. It can write the current incumbent paths atomically and emit compact
   machine-readable trace points containing elapsed time and sum of costs.
4. Its cutoff clock begins before instance construction, so the native time
   limit covers the complete C++ invocation.
5. Its command-line interface exposes the initial-path, output-path, seed and
   trace options required by the Python wrapper.
6. Its CMake configuration uses modern target-based Boost and Eigen linking,
   and the PIBT Eigen include is platform-independent.

MAPF-LNS includes rule-based PIBT, PPS and winPIBT code originating from
[Kei18/pibt](https://github.com/Kei18/pibt). Those files are covered by the
separate 2019 Keisuke Okumura MIT notice reproduced in the MAPF-LNS licence.

If this solver is used in academic work, cite:

> Jiaoyang Li, Zhe Chen, Daniel Harabor, Peter J. Stuckey and Sven Koenig.
> “Anytime Multi-Agent Path Finding via Large Neighborhood Search.” IJCAI,
> 2021.

## LaCAM3

- Upstream: [Kei18/lacam3](https://github.com/Kei18/lacam3)
- Base revision: pybind branch commit
  `c5ba7012980222489447b7d6425ce2037983c317`
- Copyright: © 2024 National Institute of Advanced Industrial Science and
  Technology (AIST)
- Licence: MIT
- Bundled licence:
  [`src/mapf_anytime/solvers/native/lacam/LICENCE.txt`](src/mapf_anytime/solvers/native/lacam/LICENCE.txt)

The bundled `lacam3/` solver core is unchanged from that upstream revision.
The local integration removes the unused command-line `argparse` target,
simplifies the pybind build, disables interprocedural optimisation for the
extension, and refactors the Python binding. The binding also exposes an
`anytime` argument that enables or disables LaCAM* refinement while preserving
LaCAM's native solution-feasibility check.

If this solver is used in academic work, cite:

> Keisuke Okumura. “Engineering LaCAM*: Towards Real-Time, Large-Scale, and
> Near-Optimal Multi-Agent Pathfinding.” AAMAS, 2024.
