public class Main {
	static Predicate isAlpha(char a) {
		return (String value) -> {
			return value.charAt(0) == a;
		};
	}

	static Parser item = new Item();
	static Parser digit = new SAT(new IsDigit(), item);
	// integer: ('+' | '-') ? [0-9]+
	static Parser integer = new SEQ(
		new ZeroOrOne(new OR(new SAT(isAlpha('+'),item), new SAT(isAlpha('-'),item))),
		new OneOrMany(digit)
	);


	public static void test(String target) {
		System.out.printf("target: %s\n", target);
		Result r = integer.parse(target);
		if (r.isSucceeded()) {
			System.out.printf("recognized: %s, remaining: %s\n", r.getRecognized(), r.getRemaining());
		} else {
			System.out.printf("%s is not integer\n", target);
		}
	}

	public static void main(String[] args) {
		String[] tests = new String[] {
			"12345a","+123","-12a456"		
		};
		for (int i = 0; i < tests.length; i++) {
			test(tests[i]);
		}
	}
}
