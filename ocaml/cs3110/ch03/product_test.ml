open OUnit
open Product

let make_product_test name expect input =
	name >:: (fun _ -> assert_equal expect (product input) ~printer:string_of_int)
	
let tests = "test suite for product" >::: [
	make_product_test "empty" 0 [0];
	make_product_test "one" 10 [1;2;5];
	make_product_test "negative" (-10) [-1;2;5];
	]

let _ = run_test_tt_main tests
