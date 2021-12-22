#include <iostream>
#include <llvm/ADT/SmallVector.h>
#include "small_vector.h"

void test_small_vector() {
    llvm::SmallVector<int, 8> vec;
    for (int i = 0; i < 8; i++) {
        vec.push_back(i);
    }
    for (const int v : vec) {
        std::cout << v << "\t";
    }
    std::cout << std::endl;
}
