#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>

using namespace llvm;

TEST(DenseMapTest, BasicTest) {
  DenseMap<int, int> mp;

  // init data
  for (int i = 0; i < 100; i++) {
    mp.insert({i, i * 100});
  }
  // lookup
  for (int i = 0; i < 100; i++) {
    auto it = mp.find(i);
    ASSERT_TRUE(it != mp.end());
    ASSERT_EQ(it->second, i * 100);
    ASSERT_EQ(it->first, i);
  }
  // remove odd
  for (int i = 1; i < 100; i += 2) {
    ASSERT_TRUE(mp.erase(i));
  }
  for (int i = 1; i < 100; i += 2) {
    ASSERT_TRUE(mp.find(i) == mp.end());
  }
  for (int i = 0; i < 100; i += 2) {
    auto it = mp.find(i);
    ASSERT_TRUE(it != mp.end());
    ASSERT_TRUE(it->first == i && it->second == i * 100);
  }
}
