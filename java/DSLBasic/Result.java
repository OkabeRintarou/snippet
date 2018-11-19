public class Result {
	private String recognized;
	private String remaining;
	private boolean succeeded;

	private Result(String recognized, String remaining, boolean succeeded) {
		this.recognized = recognized;
		this.remaining = remaining;
		this.succeeded = succeeded;
	}

	public boolean isSucceeded() {
		return succeeded;
	}

	public String getRecognized() {
		return recognized;
	}

	public String getRemaining() {
		return remaining;
	}

	public static Result succeed(String recognized, String remaining) {
		return new Result(recognized, remaining, true);
	}

	public static Result fail() {
		return new Result("", "" , false);
	}

	public static Result concat(Result r1, Result r2) {
		return new Result(r1.getRecognized().concat(r2.getRecognized()), r2.getRemaining(), true);
	}
}
