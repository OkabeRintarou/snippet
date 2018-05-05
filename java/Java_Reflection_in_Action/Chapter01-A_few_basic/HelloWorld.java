public class HelloWorld {

	public void printName() {
		System.out.println(this.getClass().getName());
	}

	private static class H2 extends HelloWorld {

	}

	public static void main(String[] args) {
		(new HelloWorld()).printName();
		(new H2()).printName();
	}
}
