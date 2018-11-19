public class OR implements Parser {
	private Parser p1;
	private Parser p2;

	public OR(Parser p1, Parser p2) {
		this.p1 = p1;
		this.p2 = p2;
	}

	public Result parse(String target) {
		Result r = p1.parse(target);
		return r.isSucceeded() ? r : p2.parse(target);
	}
}
