public class ZeroOrOne implements Parser {
	private Parser p;

	public ZeroOrOne(Parser p) {
		this.p = p;
	}

	public Result parse(String target) {
		Result r = p.parse(target);
		if (!r.isSucceeded()) {
			return Result.succeed("", target);
		}
		return r;
	}
}
