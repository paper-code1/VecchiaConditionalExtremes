#include <algorithm> // for std::find
#include <iostream>
#include <vector>

int main() {
    std::vector<int> allCenters = {1, 2, 7, 4, 5};
    int localCenter = 7;

    auto it = std::find(allCenters.begin(), allCenters.end(), 3);

    if (it != allCenters.end()) {
        std::cout << "Found localCenter: " << *it << std::endl;
    } else {
        std::cout << "localCenter not found" << std::endl;
    }

    return 0;
}
