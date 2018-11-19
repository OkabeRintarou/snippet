public class Item implements Parser {
	public Result parse(String target) {
		if (target.length() > 0) {
			return Result.succeed(target.substring(0,1), target.substring(1));
		}
		return Result.fail();
	}
}
