#pragma once
#include <type_traits>
#include <variant>

template<typename T>
class Ok {
public:
    explicit constexpr Ok(T value) : value(std::move(value)) {}
    constexpr T&& take_value() { return std::move(value); }
    T value;
};

template<typename T>
class Err {
public:
    explicit constexpr Err(T value) : value(std::move(value)) {}
    constexpr T&& take_value() { return std::move(value); }
    T value;
};

template<typename T>
constexpr Ok<std::decay_t<T>>
make_ok(T&& v) {
    return Ok<std::decay_t<T>>(std::forward<T>(v));
}

template<typename T>
constexpr Err<std::decay_t<T>> 
make_err(T&& v) {
    return Err<std::decay_t<T>>(std::forward<T>(v));
}

template<typename OkT, typename ErrT>
class Result {
public:
    using VariantT = std::variant<Ok<OkT>, Err<ErrT>>;

    constexpr Result(Ok<OkT> value) : variant_(std::move(value)) {}
    constexpr Result(Err<ErrT> value) : variant_(std::move(value)) {}

    constexpr bool is_ok() const { return std::holds_alternative<Ok<OkT>>(variant_); }
    constexpr bool is_err() const { return std::holds_alternative<Err<ErrT>>(variant_); }
    
    constexpr OkT ok_value() const { return std::get<Ok<OkT>>(variant_).value; }
    constexpr ErrT err_value() const { return std::get<Err<ErrT>>(variant_).value; }

    constexpr OkT&& take_ok_value() { return std::get<Ok<OkT>>(variant_).take_value(); }
    constexpr ErrT&& take_err_value() { return std::get<Err<ErrT>>(variant_).take_value(); }
private:
    VariantT variant_;
};
