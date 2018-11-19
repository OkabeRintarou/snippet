public class Zero implements Parser {
	public Result parse(String target) {
		return Result.succeed("", target);
	}
}
