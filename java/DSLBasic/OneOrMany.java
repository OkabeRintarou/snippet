public class OneOrMany implements Parser {
	private Parser p;

	public OneOrMany(Parser p) {
		this.p = p;
	}

	public Result parse(String target) {
		Result r = p.parse(target);

		if (!r.isSucceeded()) {
			return r;
		}

		for (;;) {
			Result r2 = p.parse(r.getRemaining());
			if (!r2.isSucceeded()) {
				break;
			}
			r = Result.concat(r,r2);
		}
		return r;
	}
}
