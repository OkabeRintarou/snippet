#include <string>
#include <gtest/gtest.h>
#include <llvm/ADT/SmallString.h>

TEST(SmallStringTest, BasicTest) {
    std::string hello("hello, world");
    llvm::SmallString<256> ss(hello);
    EXPECT_TRUE(ss.find("world") != llvm::StringRef::npos);
}

TEST(SmallStringTest, ConstructTest) {
    std::string strs[] = {"hello", "world", "china"};
    llvm::SmallString<256> ss(std::initializer_list<llvm::StringRef>{strs[0], strs[1], strs[2]});
    EXPECT_EQ(ss.size(), 15);
}
