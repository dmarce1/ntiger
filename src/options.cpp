#include <ntiger/math.hpp>
#include <ntiger/options.hpp>
#include <fstream>
#include <iostream>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/async.hpp>
#include <boost/program_options.hpp>
#include <hpx/runtime/threads/run_as_os_thread.hpp>
#include <thread>

options options::global;

HPX_PLAIN_ACTION(options::set, set_options_action);

options& options::get() {
	return global;
}

void options::set(options o) {
	global = o;
	if (global.fpe) {
		enable_floating_point_exceptions();
	}
}

bool options::process_options(int argc, char *argv[]) {
//	std::thread([&]() {
		namespace po = boost::program_options;

		po::options_description command_opts("options");

		command_opts.add_options() //
		("help", "produce help message") //
		("cfl", po::value<double>(&cfl)->default_value(0.4), "CFL factor") //
		("checkpoint", po::value<std::string>(&checkpoint)->default_value(""), "checkpoint file") //
		("config_file", po::value<std::string>(&config_file)->default_value(""), "configuration file") //
		("kernel_size", po::value<double>(&kernel_size)->default_value(-1), "softening length  < 0 = variable") //
		("fgamma", po::value<double>(&fgamma)->default_value(7.0 / 5.0), "gamma for fluid gamma law") //
		("fpe", po::value<bool>(&fpe)->default_value(true), "enable floating point exceptions") //
		("global_time", po::value<bool>(&global_time)->default_value(false), "enable global time-stepping") //
		("gravity", po::value<bool>(&gravity)->default_value(true), "enable gravity") //
		("grid_size", po::value<double>(&grid_size)->default_value(1.0), "size of grid") //
		("output_freq", po::value<double>(&output_freq)->default_value(-1), "output frequency") //
		("parts_per_node", po::value<int>(&parts_per_node)->default_value(250), "maximum number of particles on a node") //
		("problem_size", po::value<int>(&problem_size)->default_value(100), "problem size") //
		("problem", po::value<std::string>(&problem)->default_value("sod"), "problem name") //
		("theta", po::value<double>(&theta)->default_value(0.35), "theta for Barnes-Hut") //
		("tmax", po::value<double>(&tmax)->default_value(1.0), "time to end simulation") //
				;

		boost::program_options::variables_map vm;
		po::store(po::parse_command_line(argc, argv, command_opts), vm);
		po::notify(vm);
		if (vm.count("help")) {
			std::cout << command_opts << "\n";
			return false;
		}
		if (!config_file.empty()) {
			std::ifstream cfg_fs { vm["config_file"].as<std::string>() };
			if (cfg_fs) {
				po::store(po::parse_config_file(cfg_fs, command_opts), vm);
			} else {
				printf("Configuration file %s not found!\n", config_file.c_str());
				return false;
			}
		}
		po::notify(vm);

		if (output_freq <= 0.0) {
			output_freq = tmax / 100.0;
		}

//	}
//	).join();
	const auto loc = hpx::find_all_localities();
	const auto sz = loc.size();
	std::vector<hpx::future<void>> futs;
	set(*this);
	for (int i = 1; i < sz; i++) {
		futs.push_back(hpx::async<set_options_action>(loc[i], *this));
	}
	hpx::wait_all(futs);
#define SHOW( opt ) std::cout << std::string( #opt ) << " = " << std::to_string(opt) << '\n';
	SHOW(cfl);
	SHOW(fgamma);
	SHOW(fpe);
	SHOW(global_time);
	SHOW(gravity);
	SHOW(kernel_size);
	SHOW(parts_per_node);
	SHOW(problem_size);
	SHOW(theta);
	SHOW(tmax);
	return true;
}
