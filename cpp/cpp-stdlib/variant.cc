#include <iostream>
#include <string>
#include <variant>
#include <vector>

struct PrintVisitor {
  void operator()(int i) { std::cout << "int: " << i << '\n'; }
  void operator()(float f) { std::cout << "float: " << f << '\n'; }
  void operator()(const std::string &s) {
    std::cout << "string: " << s << '\n';
  }
};

static void test_variant() {
  std::variant<int, float, std::string> intFloatString;
  static_assert(std::variant_size_v<decltype(intFloatString)> == 3);
  // default initialised to the first alternative
  std::visit(PrintVisitor{}, intFloatString);

  // index will show the currently used `type`
  std::cout << "index = " << intFloatString.index() << '\n';
  intFloatString = 100.0f;
  std::cout << "index = " << intFloatString.index() << '\n';
  intFloatString = "hello world";
  std::cout << "index = " << intFloatString.index() << '\n';

  // try with get_if
  if (const auto intPtr = std::get_if<int>(&intFloatString); intPtr) {
    std::cout << "intPtr: " << *intPtr << '\n';
  } else if (const auto floatPtr = std::get_if<float>(&intFloatString);
             floatPtr) {
    std::cout << "floatPtr: " << *floatPtr << '\n';
  }

  if (std::holds_alternative<int>(intFloatString)) {
    std::cout << "the variant holds an int\n";
  } else if (std::holds_alternative<float>(intFloatString)) {
    std::cout << "the variant holds a float\n";
  } else if (std::holds_alternative<std::string>(intFloatString)) {
    std::cout << "the variant holds a string\n";
  }

  // try/catch and bad_variant_access
  try {
    auto f = std::get<float>(intFloatString);
  } catch (const std::bad_variant_access &e) {
    std::cout << "our variant doesn't hold float at this moment\n";
  }
}

static void test_variant_creation() {
  std::variant<int, float> intFloat;
  std::cout << intFloat.index() << ", val: " << std::get<int>(intFloat) << '\n';

  class NotSimple {
  public:
    NotSimple(int, float) {}
  };

  std::variant<std::monostate, NotSimple, int> okInit;
  std::cout << okInit.index() << '\n';

  // pass a value
  std::variant<int, float, std::string> intFloatString{1.5f};
  std::cout << intFloatString.index()
            << ", val: " << std::get<float>(intFloatString) << '\n';

  // error: ambiguity, double might convert to int or float
  // std::variant<int, float> intFloat2{10.5};

  std::variant<long, float, std::string> longFloatString{
      std::in_place_index<1>,
      7.6,
  };
  std::cout << longFloatString.index()
            << ", val: " << std::get<float>(longFloatString) << '\n';

  // in_place for complex types
  std::variant<std::vector<int>, std::string> vecStr{
      std::in_place_index<0>,
      {1, 2, 3},
  };
  std::cout << vecStr.index()
            << ", vector size: " << std::get<std::vector<int>>(vecStr).size()
            << '\n';
  // copy-initialize from other variant
  std::variant<int, float> intFloatSecond{intFloat};
  std::cout << intFloatSecond.index()
            << ", val: " << std::get<int>(intFloatSecond) << '\n';
}

// in_place for at lease two cases:
// 1. ambiguity
// 2. efficient complex type creation
static void test_variant_in_place_creation() {
  // ambiguity
  std::variant<int, float> intFloatFirst{std::in_place_index<0>, 10.5};
  std::variant<int, float> intFloatSecond{std::in_place_type<float>, 10.5};
  // complex type
  std::variant<std::vector<int>, std::string> vecStr{
      std::in_place_index<0>,
      {1, 2, 3, 4, 5},
  };
}

// four ways to change the current value of the variant:
// 1. the assignment operator
// 2. emplace
// 3. get and then assign a new value for the current active type
// 4. a visitor
static void test_variant_change_value() {
  //
  std::variant<int, float, std::string> intFloatString{"Hello"};
  intFloatString = 10.0f;
  intFloatString.emplace<2>("Hello");
  std::get<2>(intFloatString) += ", World";
  std::cout << "current value: " << std::get<std::string>(intFloatString)
            << '\n';

  intFloatString = 10.0f;
  if (auto float_ptr = std::get_if<float>(&intFloatString); float_ptr) {
    *float_ptr *= 2.0f;
  }
  std::cout << "current value: " << std::get<1>(intFloatString) << '\n';

  auto PrintVisitor = [](const auto &t) { std::cout << t << '\n'; };
  std::visit(PrintVisitor, intFloatString);

  auto TwiceVisitor = [](auto &t) { t *= 2; };
  std::variant<int, float> intFloat{2.0f};
  std::visit(PrintVisitor, intFloat);
  std::visit(TwiceVisitor, intFloat);
  std::visit(PrintVisitor, intFloat);
}

static void test_variant_exception() {
  class ThrowingClass {
  public:
    explicit ThrowingClass(int i) {
      if (i == 0)
        throw int(10);
    }
    operator int() { throw int(10); }
  };

  std::variant<int, ThrowingClass> v;
  try {
    v = ThrowingClass(0);
  } catch (...) {
    std::cout << "catch(...)" << '\n';
    std::cout << v.valueless_by_exception() << '\n';
    std::cout << std::get<int>(v) << '\n';
  }
}

static void test_variant_sizeof() {
  std::cout << "sizeof string: " << sizeof(std::string) << '\n';
  std::cout << "sizeof variant<int, string>: "
            << sizeof(std::variant<int, std::string>) << '\n';
  std::cout << "sizeof variant<int, float>: "
            << sizeof(std::variant<int, float>) << '\n';
  std::cout << "sizeof variant<int, double>: "
            << sizeof(std::variant<int, double>) << '\n';
}

static void test_variant_fsm() {

  struct DoorState {
    struct DoorOpened {};
    struct DoorClosed {};
    struct DoorLocked {};

    using State = std::variant<DoorOpened, DoorClosed, DoorLocked>;

    struct OpenEvent {
      State operator()(const DoorOpened &) { return DoorOpened{}; }
      State operator()(const DoorClosed &) { return DoorOpened{}; }
      State operator()(const DoorLocked &) { return DoorLocked{}; }
    };

    struct CloseEvent {
      State operator()(const DoorOpened &) { return DoorClosed{}; }
      State operator()(const DoorClosed &) { return DoorClosed{}; }
      State operator()(const DoorLocked &) { return DoorLocked{}; }
    };

    struct LockEvent {
      State operator()(const DoorOpened &) { return DoorOpened{}; }
      State operator()(const DoorClosed &) { return DoorLocked{}; }
      State operator()(const DoorLocked &) { return DoorLocked{}; }
    };

    struct UnlockEvent {
      State operator()(const DoorOpened &) { return DoorOpened{}; }
      State operator()(const DoorClosed &) { return DoorClosed{}; }
      State operator()(const DoorLocked &) { return DoorClosed{}; }
    };

    void open() { state = std::visit(OpenEvent{}, state); }
    void close() { state = std::visit(CloseEvent{}, state); }
    void lock() { state = std::visit(LockEvent{}, state); }
    void unlock() { state = std::visit(UnlockEvent{}, state); }

    struct PrintVisitor {
      void operator()(const DoorOpened &) { std::cout << "opened\n"; }
      void operator()(const DoorClosed &) { std::cout << "closed\n"; }
      void operator()(const DoorLocked &) { std::cout << "locked\n"; }
    };
    void print() { std::visit(PrintVisitor{}, state); }
    State state;
  };

  DoorState door_state;
  door_state.open();
  door_state.print();
  door_state.lock();
  door_state.print();
  door_state.close();
  door_state.lock();
  door_state.print();
}

int main() {
  test_variant();
  std::cout << std::string(35, '-') << '\n';
  test_variant_creation();
  std::cout << std::string(35, '-') << '\n';
  test_variant_in_place_creation();
  std::cout << std::string(35, '-') << '\n';
  test_variant_change_value();
  std::cout << std::string(35, '-') << '\n';
  test_variant_exception();
  std::cout << std::string(35, '-') << '\n';
  test_variant_sizeof();
  std::cout << std::string(35, '-') << '\n';
  test_variant_fsm();
  return 0;
}