public class IsDigit implements Predicate {
	public boolean satisfy(String value) {
		char c = value.charAt(0);
		return c >= '0' && c <= '9';
	}
}
