public class SEQ implements Parser {
	private Parser p1;
	private Parser p2;

	public SEQ(Parser p1, Parser p2) {
		this.p1 = p1;
		this.p2 = p2;
	}

	public Result parse(String target) {
		Result r1 = p1.parse(target);
		if (r1.isSucceeded()) {
			Result r2 = p2.parse(r1.getRemaining());
			if (r2.isSucceeded()) {
				return Result.concat(r1,r2);
			}
		}
		return Result.fail();
	}
}
