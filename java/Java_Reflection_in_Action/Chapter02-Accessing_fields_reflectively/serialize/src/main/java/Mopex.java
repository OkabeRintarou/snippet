import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.LinkedList;
import java.util.List;

public class Mopex {
    public static Field[] getInstanceVariables(Class cls) {
        List accum = new LinkedList();
        while (cls != null) {
            Field[] fields = cls.getDeclaredFields();
            for (Field field : fields) {
                if (!Modifier.isStatic(field.getModifiers())) {
                    accum.add(field);
                }
            }
            cls = cls.getSuperclass();
        }
        return (Field[]) accum.toArray(new Field[accum.size()]);
    }
}
