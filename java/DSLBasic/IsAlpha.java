public class IsAlpha implements Predicate {
	public boolean satisfy(String value) {
		char c = value.charAt(0);
		return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
	}
}
