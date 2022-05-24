[TOC]

# Creation

```c++
// default
std::variant<int, float> intFloat;
std::variant<std::monostate, int, float> okInit;

// pass a value
std::variant<int, float, std::string> intFloatString {10.5f};

// ambiguity resolved by in_place
std::variant<long, float, std::string> longFloatString {
	std::in_place_index<1>, 7.6,  
};
```

# Change

emplace

swap

# Accessing

std::get<Type | Index>: throw bad_variant_access exception

std::get_if: return nullptr on error