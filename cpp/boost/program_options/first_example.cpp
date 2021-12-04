#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace po = boost::program_options;
using namespace std;

ostream &operator<<(ostream &os, const vector<string> &vec) {
    for (const auto &s : vec) {
        os << s << "\t";
    }
    return os;
}

int main(int argc, char *argv[]) {
    int opt;
    string config_file;
    // Declare a group of options will be
    // allowed only on command line
    po::options_description generic("Generic options");

    generic.add_options()("version,v", "print version message")(
        "help",
        "produce help message")("config,c",
                                po::value<string>(&config_file)
                                    ->default_value("multiple_sources.cfg"),
                                "name of a file of a configuration");

    po::options_description config("Configuration");
    config.add_options()("optimization",
                         po::value<int>(&opt)->default_value(10),
                         "optimization level")(
        "include-path,I", po::value<vector<string>>(), "include path");

    po::options_description hidden("Hideen options");
    hidden.add_options()("input-file", po::value<vector<string>>(),
                         "input file");

    po::options_description cmdline_options;
    cmdline_options.add(generic).add(config).add(hidden);

    po::options_description config_file_options;
    config_file_options.add(config).add(hidden);

    po::options_description visible("Allowed options");
    visible.add(generic).add(config);

    po::positional_options_description p;
    p.add("input-file", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                  .options(cmdline_options)
                  .positional(p)
                  .run(),
              vm);
    po::notify(vm);

    ifstream ifs(config_file.c_str());
    if (!ifs) {
        cout << "can not open config file: " << config_file << "\n";
        return 0;
    } else {
        po::store(parse_config_file(ifs, config_file_options), vm);
        po::notify(vm);
    }

    if (vm.count("help")) {
        std::cout << visible << "\n";
        return 1;
    }
    if (vm.count("version")) {
        cout << "Multiple sources example, version 1.0\n";
        return 0;
    }

    if (vm.count("include-path")) {
        cout << "Include paths are: " << vm["include-path"].as<vector<string>>()
             << "\n";
    }

    if (vm.count("input-file")) {
        cout << "input files are: " << vm["input-file"].as<vector<string>>()
             << "\n";
    }

    cout << "Optimization level is " << opt << "\n";

    return 0;
}
