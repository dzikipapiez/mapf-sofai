#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include "LNS.h"
#include "AnytimeBCBS.h"
#include "AnytimeEECBS.h"
#include "PIBT/pibt.h"


/* Main function */
int main(int argc, char** argv)
{
	namespace po = boost::program_options;
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")

		// params for the input instance and experiment settings
		("map,m", po::value<string>()->required(), "input file for map")
		("agents,a", po::value<string>()->required(), "input file for agents")
		("agentNum,k", po::value<int>()->default_value(0), "number of agents")
        ("seed", po::value<int>()->default_value(0), "random seed for MAPF-LNS")
        ("output,o", po::value<string>(), "output file")
		("cutoffTime,t", po::value<double>()->default_value(7200), "cutoff time (seconds)")
		("screen,s", po::value<int>()->default_value(0),
		        "screen option (0: none; 1: LNS results; 2:LNS detailed results; 3: MAPF detailed results)")
		("stats", po::value<string>(), "output stats file")
        ("initPaths", po::value<string>(), "external initial solution path file for LNS")
        ("paths", po::value<string>(), "output final paths file")
        ("trace", po::bool_switch()->default_value(false), "print machine-readable LNS runtime/SoC trace lines")
        ("earlyStopSeconds", po::value<double>(), "stop LNS after this many seconds without improvement")

		// solver
		("solver", po::value<string>()->default_value("LNS"), "solver (LNS, A-BCBS, A-EECBS)")

        // params for LNS
        ("neighborSize", po::value<int>()->default_value(5), "Size of the neighborhood")
        ("maxIterations", po::value<int>()->default_value(1000000), "maximum number of iterations")
        ("replanTimeLimit", po::value<double>()->default_value(1.0),
                "maximum time for one neighborhood replanning attempt")
        ("initAlgo", po::value<string>()->default_value("EECBS"),
                "MAPF algorithm for finding the initial solution (EECBS, PP, PPS, CBS, PIBT, winPIBT)")
        ("replanAlgo", po::value<string>()->default_value("PP"),
                "MAPF algorithm for replanning (EECBS, CBS, PP)")
        ("destoryStrategy", po::value<string>()->default_value("Adaptive"),
                "Heuristics for finding subgroups (Random, RandomWalk, Intersection, Adaptive)")
        ("pibtWindow", po::value<int>()->default_value(5),
             "window size for winPIBT")
        ("winPibtSoftmode", po::value<bool>()->default_value(true),
             "winPIBT soft mode")
		;
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);

	if (vm.count("help")) {
		cout << desc << endl;
		return 1;
	}

    PIBTPPS_option pipp_option;
    pipp_option.windowSize = vm["pibtWindow"].as<int>();
    pipp_option.winPIBTSoft = vm["winPibtSoftmode"].as<bool>();

    po::notify(vm);

    const auto budget_start = Time::now();
	Instance instance(vm["map"].as<string>(), vm["agents"].as<string>(),
		vm["agentNum"].as<int>());
    double time_limit = vm["cutoffTime"].as<double>();
    int screen = vm["screen"].as<int>();
	srand(vm["seed"].as<int>());

    if (vm["solver"].as<string>() == "LNS")
    {
        if (vm["replanTimeLimit"].as<double>() <= 0)
        {
            cerr << "replanTimeLimit must be positive." << endl;
            return 1;
        }
        LNS lns(instance, time_limit, budget_start,
                vm["initAlgo"].as<string>(),
                vm["replanAlgo"].as<string>(),
                vm["destoryStrategy"].as<string>(),
                vm["neighborSize"].as<int>(),
                vm["maxIterations"].as<int>(),
                vm["replanTimeLimit"].as<double>(), screen, pipp_option);
        lns.setTrace(vm["trace"].as<bool>());
        if (vm.count("earlyStopSeconds"))
        {
            const auto seconds = vm["earlyStopSeconds"].as<double>();
            if (seconds <= 0)
            {
                cerr << "earlyStopSeconds must be positive." << endl;
                return 1;
            }
            lns.setEarlyStopSeconds(seconds);
        }
        if (vm.count("paths"))
            lns.setIncumbentPathFile(vm["paths"].as<string>());
        if (vm.count("initPaths"))
        {
            if (!lns.loadInitialSolution(vm["initPaths"].as<string>()))
                return -1;
        }
        bool succ = lns.run();
        if (succ)
            lns.validateSolution();
        if (vm.count("output"))
            lns.writeResultToFile(vm["output"].as<string>());
        if (vm.count("stats"))
            lns.writeIterStatsToFile(vm["stats"].as<string>());
        if (succ && vm.count("paths"))
            lns.writePathsToFile(vm["paths"].as<string>());
    }
    else if (vm["solver"].as<string>() == "A-BCBS") // anytime BCBS(w, 1)
    {
        AnytimeBCBS bcbs(instance, time_limit, screen);
        bcbs.run();
        bcbs.validateSolution();
        if (vm.count("output"))
            bcbs.writeResultToFile(vm["output"].as<string>());
        if (vm.count("stats"))
            bcbs.writeIterStatsToFile(vm["stats"].as<string>());
    }
    else if (vm["solver"].as<string>() == "A-EECBS") // anytime EECBS
    {
        AnytimeEECBS eecbs(instance, time_limit, screen);
        eecbs.run();
        eecbs.validateSolution();
        if (vm.count("output"))
            eecbs.writeResultToFile(vm["output"].as<string>());
        if (vm.count("stats"))
            eecbs.writeIterStatsToFile(vm["stats"].as<string>());
    }
	else
    {
	    cerr << "Solver " << vm["solver"].as<string>() << " does not exist!" << endl;
	    exit(-1);
    }
	return 0;

}
