/**
 * VM Args: -verbose:gc -client -Xms20M -Xmx20M -Xmn10M -XX:+PrintGCDetails
 * -XX:+UseSerialGC -XX:SurvivorRatio=8 -XX:MaxTenuringThreshold=15
 */
public class TestTenuringThreshold2 {
  private static final int _1M = 1024 * 1024;

  public static void testTenuringThreshold2() {
    byte[] a1, a2, a3, a4;
    a1 = new byte[_1M / 4];
    a2 = new byte[_1M / 4];
    // a1 + a2 大于survivo空间一半
    a3 = new byte[4 * _1M];
    a4 = new byte[4 * _1M];
    a4 = null;
    a4 = new byte[4 * _1M];
  }

  public static void main(String[] args) { testTenuringThreshold2(); }
}
