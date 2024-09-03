#include <iostream>
#include <string>
#include <vector>
#include "random_points.h"

void print_usage(const std::string& program_name) {
    std::cerr << "Usage: " << program_name << " --num_loc <number_of_locations> --sub_partition <num_blocks_x> <num_blocks_y> [--print]" << std::endl;
}

bool parse_args(int argc, char** argv, Options& opts) {
    if (argc < 6) {
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--num_loc") {
            if (i + 1 < argc) {
                opts.numPointsPerProcess = std::atoi(argv[++i]);
            } else {
                return false;
            }
        } else if (arg == "--sub_partition") {
            if (i + 2 < argc) {
                opts.numBlocksX = std::atoi(argv[++i]);
                opts.numBlocksY = std::atoi(argv[++i]);
            } else {
                return false;
            }
        } else if (arg == "-m") {
            if (i + 1 < argc) {
                opts.m = std::atoi(argv[++i]);
            } else {
                return false;
            }
        } else if (arg == "--print") {
            opts.print = true;
        } else {
            return false;
        }
    }

    return true;
}
