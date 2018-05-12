/*
 * VM Args: -XX:PermSize=10M -XX:MaxPermSize=10M
 */
import java.util.List;
import java.util.ArrayList;

public class RuntimeConstantPoolOOM {
  public static void main(String[] args) {
    List<String> list = new ArrayList<String>();
    int i = 0;
    for (;;) {
      list.add(String.valueOf(i++).intern());
    }
  }
}
