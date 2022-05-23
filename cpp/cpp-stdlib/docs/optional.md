[TOC]
# Creation

```c++
// empty
std::optional<int> oEmpty;

// direct
std::optional<int> oInt = 10;
std::optional oIntDeduced(10);

// make_optional
auto oDouble = std::make_optional(3.0);

// in_place
std::optional<std::complex<double>> o7{std::in_place, 3.0, 4.0};
std::optional<std::vector<int>> oVec(std::place, {1, 2, 3});

// copy from other optional
auto oIntCopy = oInt;
```

对于 non copyable/moveable 类，只有通过 std::in_place 构造出 std::option 对象

# Return

当函数返回值类型是 std::optional 时，要注意当返回值被包在 \{\} 中时会阻止移动构造，返回对象将被拷贝构造

```c++
std::optional<std::string> CreateString() {
    std::string str{"Hello, World!"};
    return {str};	// this one will cause a copy
    // return str;	// this one moves
}
```

# Accessing

operator\* and operator\-> : if there's no value the behaviour is **undefined**

value(): returns the value of throws std::bad_optional_access

value_or(default_val): returns the value if available, or default_val otherwise

# Change

swap()

reset()

emplace()

# Reference

[Optional Examples Wall](https://www.cppstories.com/2018/06/optional-examples-wall/)
