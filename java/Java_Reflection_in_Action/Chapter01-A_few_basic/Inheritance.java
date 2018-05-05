import java.lang.reflect.*;

public class Inheritance {
	public static void main(String[] args) {
		System.out.println(Object.class.isAssignableFrom(String.class)); // true
		System.out.println(java.util.List.class.isAssignableFrom(java.util.Vector.class)); // true
		System.out.println(double.class.isAssignableFrom(double.class)); // true
		System.out.println(Object.class.isAssignableFrom(double.class)); // false

		System.out.println(Class.class.isInstance(Object.class)); // true
		System.out.println(Object.class.isAssignableFrom(Class.class)); // true
		System.out.println(Class.class.isInstance(Class.class)); // true, Class is Java's only metaclass
	}
}
