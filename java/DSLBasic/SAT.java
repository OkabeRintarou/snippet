public class SAT implements Parser {
	private Predicate pre;
	private Parser parser;

	public SAT(Predicate pre, Parser parser) {
		this.pre = pre;
		this.parser = parser;
	}

	public Result parse(String target) {
		Result r = parser.parse(target);
		if (r.isSucceeded() && pre.satisfy(r.getRecognized())) {
			return r;
		}
		return Result.fail();
	}
}
