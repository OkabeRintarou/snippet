#pragma once

class Test {
public:
  Test();
  ~Test();

  Test(const Test&);
  Test& operator=(const Test&);
  Test(Test &&);
  Test& operator=(Test&&);
};
