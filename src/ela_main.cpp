// https://stackoverflow.com/questions/45466939/parsing-command-line-arguments-with-boost-program-options-c
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <exception>

//using namespace std;
using std::cerr;
using std::cout;
using std::endl;
using std::exception;
//using std::vector;
using std::string;

#include "version.h"
#include "param.h"
#include "stations.h"
#include "layers.h"
#include "hypfile.h"
#include "arrivals.h"
#include "gather.h"
#include "output_hy3.h"
#include "hy3file.h"


#include <cstdlib>
#include <string>
std::string get_env_var(const std::string &var_name, const std::string &default_value = "") {
    // Use C function getenv to retrieve the environment variable
    // var_name: name of the environment variable
    // default_value: value to return if the environment variable is not found
    const char *value = std::getenv(var_name.c_str());
    if (value) {
	return std::string(value);
    } else {
	return default_value;
    }
}

int get_hypo(const CArrivalFile & arrs, const TParams &param,
	       std::array<double,4> & hypo, double (& covM)[4][4]);

int main(int ac, char** av){

string file_i;
string file_o;
string file_m;
string file_s;
double reading_err;
double model_err;
string config_file;

bool option_missing(false);

// print version
cout << "ELA " << getVersion() << endl;

// parse command line options
try {

    po::options_description desc("Usage: ela [options] input \n" \
		    "Allows options:");
    
    po::options_description generic("Generic options");
    generic.add_options()
	("version,v", "print version string")
        ("help,h", "help message")
	("config,c",
	 po::value<string>(&config_file)->default_value("ela.cfg"),
	 "configuration file (full path)");
	
    po::options_description cmd("Command line I/O options");
    cmd.add_options()
        ("input,i", po::value<string>(), "input hyp file")
        ("output,o", po::value<string>(), "output hy3 file");

    po::options_description config("Configuration");
    config.add_options()
        ("model,m", po::value<string>(), "input model file")
        ("stations,s", po::value<string>(), "input list of stations")
        ("reading_err", po::value<double>(&reading_err)->default_value(0),
	 "set arrival time reading error [s]")
        ("model_err", po::value<double>(&model_err)->default_value(0),
	 "set propagation time prediction error [s]");

    desc.add(generic).add(cmd).add(config);

    po::options_description config_file_options;
    config_file_options.add(cmd).add(config);

    po::variables_map vm;
//    po::store(po::parse_command_line(ac, av, desc), vm);


    po::positional_options_description p;
    p.add("input", -1);

    po::store(po::command_line_parser(ac, av).
          options(desc).positional(p).run(), vm);
    po::notify(vm);


    if (vm.count("help")) {
        cout << desc << "\n";
        return 0;
      }

    if (vm.count("version")) {
        cout << "ELA " << getVersion() << " | Date: " << getBuildDate() << " | Time: " << getBuildTime() << endl;
        return 0;
      }

    // read configuration file options
    // Path to configuration file is in config_file variable set above.
    // config_file is set to full path if -c option is used
    // or set to 'ela.cfg' in current directory by default.
    // If 'ela.cfg' file is not found in current directory,
    // read ela.cfg from ELA_CONFIG directory.
    std::ifstream ifs(config_file.c_str());
    if (!ifs) {
	// if file was not found
	// and config_file contains only file name 'ela.cfg'
	// try to read from ELA_CONFIG directory
	if (config_file == "ela.cfg") {
	    std::string env_config_dir = get_env_var("ELA_CONFIG",
			    "/opt/ipe/share/ela");
	    if (!env_config_dir.empty()) {
		config_file = env_config_dir + "/" + config_file;
		ifs.open(config_file.c_str());
	    }
	    else {
	        // if ELA_CONFIG is not set, print error and exit.
		cout << "Environment variable ELA_CONFIG is not set.\n"
		     << "Cannot find configuration file: " << config_file << "\n";
		return 1;
	    }
	}
    }

    if (ifs) {
	po::store(po::parse_config_file(ifs, config_file_options), vm);
	po::notify(vm);
      }
    else {
        cout << "Cannot open config file: " << config_file << "\n";
        return 1;
      }

    // input hyp file
    if (vm.count("input")) {
	file_i = vm["input"].as<string>();
        cout << "Input file: " << file_i << "\n";
        }
    else {
        cout << "Input file was not set (option -i).\n";
	option_missing=true;
        }

    // output hy3 file
    if (vm.count("output")) {
        file_o = vm["output"].as<string>();
        cout << "Output file: " << file_o << "\n";
        }
    else if (vm.count("input")) {
        file_o = vm["input"].as<string>();
	file_o = file_o.substr(file_o.find_last_of("/")+1,file_o.npos); 
	file_o = "./ela_"+file_o.substr(0, file_o.find_last_of("."))+".hy3"; 
        cout << "Output file: " << file_o << "\n";
        }
    else {
        cout << "Output file was not set (option -o).\n";
	option_missing=true;
        }

    // velocity model file
    if (vm.count("model")) {
	file_m = vm["model"].as<string>();
        cout << "Velocity model file: " << file_m << "\n";
        }
    else {
        cout << "Velocity model file was not set (option -m).\n";
	option_missing=true;
        }

    // stations list file
    if (vm.count("stations")) {
        file_s = vm["stations"].as<string>();
        cout << "Stations list file: " << file_s << "\n";
        }
    else {
        cout << "Stations list was not set (option -s).\n";
	option_missing=true;
        }

    if (vm.count("reading_err")) {
        cout << "Reading error: "
             << vm["reading_err"].as<double>() << "\n";
        }

    if (vm.count("model_err")) {
        cout << "Model error: "
             << vm["model_err"].as<double>() << "\n";
        }

    } //try

    catch(exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    return 1;
    }

    if (option_missing) {return 1;}

struct TParams param;

param.reading_err=reading_err;
param.model_err=model_err;
param.name_model=file_m.substr(file_m.find_last_of("/")+1,file_m.npos);
param.name_event=file_i.substr(file_i.find_last_of("/")+1,file_i.npos);
param.name_event=param.name_event.substr(0, param.name_event.find_last_of("."));


    // initial point
    std::array<double,4> startpt{0.0};
    bool init_nea = false;

do {
    // start point
    startpt = {0.0, 0.0, 0.0, 0.0};
    std::cout << " Start point [X,Y,Z]\n"
              << "       orig. time given by minimizing procedure\n"
              << "       space coord. = 0 ... value of the nearest station\n"
              << "       [0,0,0]:_";
    std::string line;
    if (!std::getline(std::cin, line)) {
        continue;
    }
    if (line.empty()) {
        init_nea = true;
	break;
    } else {
	// numbers are separated by spaces or commas
	if (line.find(',') != std::string::npos) {
	    // replace commas with spaces
	    std::replace(line.begin(), line.end(), ',', ' ');
	}
        std::istringstream iss(line);
        // Attempt to read 4 values into startpt.
        if (!(iss >> startpt[0] >> startpt[1] >> startpt[2])) {
            continue; // Error: not enough values or invalid input
		      // continue to prompt again
	}
	/*
        if (!(iss >> startpt[3])) {
	    startpt[3] = 0.0; // Default value for the 4th element
	}
	*/

	break; // exit the loop
    }
} while (true);
	
    bool fix_x = false;
    bool fix_y = false;
    bool fix_depth = false;

do {
    // fixed coordinates of the epicenter
    fix_x = false;
    fix_y = false;
    fix_depth = false;
    std::cout << " Enter fixed coordinates of the epicenter:\n"
              << "         Fixed coordinates XYZ   -  ''X or Y or XYZ''\n"
              << "         Fixed depth             -  ''Z''   [ ]:_";
    std::string line;
    if (!std::getline(std::cin, line)) {
        continue;
    }
    if (line.empty()) {
        break; // default values
    } else {
        // Check for 'X' or 'x'
        if (line.find('X') != std::string::npos || line.find('x') != std::string::npos) {
            fix_x = true;
	    break;
        }
        // Check for 'Y' or 'y'
        if (line.find('Y') != std::string::npos || line.find('y') != std::string::npos) {
            fix_x = true;
	    break;
        }
        // Check for 'Z' or 'z'
        if (line.find('Z') != std::string::npos || line.find('z') != std::string::npos) {
            fix_depth = true;
	    break;
        }
    }
} while (true);

param.fix = {fix_x, fix_y, fix_depth, false};
param.init_nea = init_nea;   
param.startpt = startpt;

// if X o Y coordinates are fixed, then all X, Y, Z coordinates are fixed
if (param.fix[0] || param.fix[1]) {
	param.fix[0] = true;
	param.fix[1] = true;
	param.fix[2] = true;
}


std::cout << " Event name: " << param.name_event << "\n";
std::cout << " Model name: " << param.name_model << "\n";
std::cout << " Model error: " << param.model_err << "\n";
std::cout << " Reading error: " << param.reading_err << "\n";
std::cout << " Start point: " << param.startpt[0] << " "
	  << param.startpt[1] << " " << param.startpt[2] << " "
	  << param.startpt[3] << "\n";
std::cout << " Fix X: " << param.fix[0] << "\n";
std::cout << " Fix Y: " << param.fix[1] << "\n";
std::cout << " Fix Z: " << param.fix[2] << "\n";
std::cout << " Fix T: " << param.fix[3] << "\n";
std::cout << " Init NEA: " << param.init_nea << "\n";



    // read hyp file and fill 'hyp' - vector of TRecordHyp
//    std::map<string, TStation> stations;
//    std::map<std::string, std::shared_ptr<TLayers>> p_layers_map;
//    CArrivalFile arrs;
//    std::vector<TRecordHyp> hyp;

    // read stations and fill 'stations' - map of TStation
    loadStations(file_s);
    // printStations();

    // read model and fill 'p_layers_map' - map of TLayers
    loadLayersPointerMap(file_m);
    // printLayersPointerMap();

    // read hyp file and fill 'hyp' - vector of TRecordHyp
    loadHyp(file_i);
    // printHyp();

    // fill 'arrs' - vector of TArrival
    arrs.fill(hyp, stations);
    
    std::array<double,4> hypo{0.0};
    static double covM[4][4] = {0.0};
    // locate the hypocenter hypo and covariance matrix covM
    int info = get_hypo(arrs, param, hypo, covM);
    std::cout << "     EXIT PARAMETER" << std::setw(16) << "" << std::setw(10) << info << std::endl;
    std::cout << "     FINAL APPROXIMATE SOLUTION";
    for (size_t i = 0; i < hypo.size(); i++) {
	    std::cout << std::setw(15) << std::fixed << std::setprecision(7) << hypo[i];
    }
    std::cout << std::endl;
    
    struct hy3_file hy3;
    int nrec = arrs.n_arr;
    hy3.rec = (struct hy3_record *) malloc (nrec * sizeof(struct hy3_record));
    output_hy3(hy3, param, hypo.data(), arrs, hyp, gather, (double*) covM, info); 
    //hy3print (&hy3);
    
    // write to file using hy3save
    FILE *fout;
    fout = fopen(file_o.c_str(), "wt");
    if (fout == NULL) {
	std::cerr << "Error opening file for writing: " << file_o << std::endl;
	return 1;
    }
    
    hy3save(&hy3, fout);


return 0;
}
