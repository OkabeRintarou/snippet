open OUnit
open Library

let make_nth_test name expected func input =
	name >:: (fun _ -> assert_equal expected (func input))
 
let nth_tests = "test suite for library" >::: [
		make_nth_test "empty" 0 fifth [];
		make_nth_test "one" 0 fifth [0];
		make_nth_test "two" 0 fifth [0;1];
		make_nth_test "three" 0 fifth [0;1;2];
		make_nth_test "four" 0 fifth [0;1;2;3];
		make_nth_test "five" 4 fifth [0;1;2;3;4];
		make_nth_test "ten" 4 fifth [0;1;2;3;4;5;6;7;8;9];
	]

let _ = run_test_tt_main nth_tests

