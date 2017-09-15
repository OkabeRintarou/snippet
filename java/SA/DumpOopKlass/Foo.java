public class Foo {
	public static int foo_instance_i = 7777;
	
	public Foo(int i) {
		foo_instance_i = i;
	}

	public int getInstance_i() {
		return foo_instance_i;
	}
}
