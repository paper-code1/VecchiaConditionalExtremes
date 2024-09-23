#include <iostream>
#include <string>
#include <vector>

#include "cxxopts.hpp"

// Structure to store command line options
struct Opts
{
    int numPointsPerProcess;
    int numBlocksX;
    int numBlocksY;
    int m; // the number of nearest neighbor
    bool print;
};

bool parse_args(int argc, char **argv, Opts &opts)
{
    cxxopts::Options options(argv[0], "Block Vecchia approximation");
    options.add_options()
    ("num_loc_per_process", "Number of locations for each processor", cxxopts::value<int>()->default_value("2000"))
    ("sub_partition", "Number of blocks in x and y directions (format: x,y)", cxxopts::value<std::vector<int>>()->default_value("2,40"))
    ("print", "Print additional information", cxxopts::value<bool>()->default_value("false"))
    ("m", "Special rule for the first 100 blocks", cxxopts::value<int>()->default_value("30"))
    ("help", "Print usage");

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return false;
    }

    opts.numPointsPerProcess = result["num_loc_per_process"].as<int>();
    
    auto sub_partition = result["sub_partition"].as<std::vector<int>>();
    if (sub_partition.size() == 2)
    {
        opts.numBlocksX = sub_partition[0];
        opts.numBlocksY = sub_partition[1];
    }
    else
    {
        std::cerr << "Error: --sub_partition requires exactly two values and its format should be (x,y)" << std::endl;
        return false;
    }
    opts.print = result["print"].as<bool>();
    opts.m = result["m"].as<int>();

    return true;
}