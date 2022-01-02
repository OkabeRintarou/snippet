#include <string>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/ArrayRef.h>
#include <gtest/gtest.h>

void read_small_vector(llvm::ArrayRef<int> array) {
    for (auto i = 0; i < array.size(); i++) {
        EXPECT_EQ(array[i], i);
    }
}

void write_small_vector(llvm::SmallVectorImpl<int> &vec) {
    for (int i = 0; i < 8; i++) {
        vec.push_back(i);
    }
}

TEST(SmallVectorTest, BasicTest) {
    llvm::SmallVector<int, 8> vec;
    for (int i = 0; i < 8; i++) {
        vec.emplace_back(i);
    }
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(vec[i], i);
    }
    read_small_vector(vec);
    write_small_vector(vec);
    EXPECT_EQ(vec.size(), 16);

    vec.clear();
    EXPECT_TRUE(vec.empty());
}