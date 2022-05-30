#pragma once

#include <type_traits>
#include <variant>

template<typename OkT, typename ErrT>
class Result {
private:
  using VariantT = std::variant<OkT, ErrT>;
  template<typename T, typename U>
  friend Result<T, U> make_ok(T);
  template<typename T, typename U>
  friend Result<T, U> make_err(U);

  Result(const Result &) = delete;
  void operator=(const Result&) = delete;
public:
  Result() = default;

  Result(Result &&o) noexcept : variant_(std::move(o.variant_)) {}
  Result& operator=(Result &&o) noexcept {
    if (this != &o) {
      variant_ = std::move(o.variant_);
    }
    return *this;
  }

  constexpr operator bool() const { return is_ok(); }
  constexpr bool is_ok() const { return variant_.index() == 0; }
  constexpr bool is_err() const { return variant_.index() == 1; }

  constexpr const OkT &ok_value() const { return std::get<0>(variant_); }
  constexpr const ErrT &err_value() const { return std::get<1>(variant_); }
  constexpr OkT&& take_ok_value() { return std::get<0>(std::move(variant_)); }
  constexpr ErrT&& take_err_value() { return std::get<1>(std::move(variant_)); }
private:
  VariantT variant_;
};

template<typename OkT, typename ErrT>
Result<OkT, ErrT> make_ok(OkT value) {
  Result<OkT, ErrT> r;
  r.variant_.template emplace<0>(std::move(value));
  return r;
}

template<typename OkT, typename ErrT>
Result<OkT, ErrT> make_err(ErrT value) {
  Result<OkT, ErrT> r;
  r.variant_.template emplace<1>(std::move(value));
  return r;
}


