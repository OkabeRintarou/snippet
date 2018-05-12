/*
 * VM Args: -Xms20m -Xmx20m -XX:+HeapDumpOnOutOfMemoryError
 */
import java.util.List;
import java.util.ArrayList;

public class HeapOOM {
  static class OOMObject {}

  public static void main(String[] args) {
    List<OOMObject> list = new ArrayList<OOMObject>();

    for (;;) {
      list.add(new OOMObject());
    }
  }
}
