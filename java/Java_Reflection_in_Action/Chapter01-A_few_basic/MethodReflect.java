import java.lang.reflect.Method;
import java.util.Vector;
import java.util.Collection;
import java.util.Arrays;

public class MethodReflect {
	public static void main(String[] args) throws Exception {
		Class intClass = int.class;
		Class interfaceClass = Collection.class;
		Class arrayClass = Object[].class;
		Method methodGet = Vector.class.getMethod("get",intClass);
		Method methodAddAll = Vector.class.getMethod("addAll",interfaceClass);
		Method methodCoptInto = Vector.class.getMethod("copyInto",arrayClass);
		System.out.println("int isPrimitive: " + ( intClass.isPrimitive() ? "true" : "false"));
		System.out.println("Collection isInterface: " + (interfaceClass.isInterface() ? "true" : "false"));
		System.out.println("Object[] isArray: " + (arrayClass.isArray() ? "true" : "false"));

		// Method.invoke(Object obj,Object[] args)
		Vector<String> v = new Vector<String>(Arrays.asList("C++","Java","Python"));
		String item1 = (String)(methodGet.invoke(v,1)); 
		System.out.println("element at index 1: " + item1);

		// call method with no argument through reflection
		Method methodHashCode = v.getClass().getMethod("hashCode",new Class[0]);
		int code = ((Integer)(methodHashCode.invoke(v,new Object[]{}))).intValue();
		System.out.printf("hashcode: %x\n",code);
	}
}
