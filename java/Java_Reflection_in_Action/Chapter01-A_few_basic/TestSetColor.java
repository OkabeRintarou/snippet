import java.lang.reflect.*;

public class TestSetColor {
	public static Method getSupportedMethod(Class cls,String name,Class... paramTypes) throws NoSuchMethodException {
		if (cls == null) {
			throw new NoSuchMethodException();
		}
		try {
			return cls.getDeclaredMethod(name,paramTypes);
		} catch (NoSuchMethodException e) {
			return getSupportedMethod(cls.getSuperclass(),name,paramTypes);
		}
	}

	public static void setObjectColor(Object obj,Color color) {
		Class cls = obj.getClass();
		try {
			Method method = getSupportedMethod(cls,"setColor",Color.class);
			method.invoke(obj,color);
		} catch (NoSuchMethodException e) {
			throw new IllegalArgumentException(
						cls.getName() + " does not support"
						+ " method setColor(:Color)");
		} catch (IllegalAccessException e) {
			throw new IllegalArgumentException(
						"Insufficient access permissions to call"
						+ " setColor(:Color) in class "
						+ cls.getName());
		} catch (InvocationTargetException e) {
			throw new RuntimeException(e);
		}
	}

	public static void main(String[] args) {
		Shape shape = new Shape();
		Animal animal = new Animal();
		Color color = new Color();
		try {
			setObjectColor(shape,color);
		} catch (Exception e) {
			System.out.println(e);
		}

		try {
			setObjectColor(animal,color);
		} catch (Exception e) {
			System.out.println(e);
		}
	}
}


class Color {
}

class Shape {
	private void setColor(Color color) {
		System.out.println("Shape.setColor");
	}
}

class Animal {
	public void setColor(Color color) {
		System.out.println("Animal setColor");
	}
}
