/**
 * VM Args: -verbose:gc -client -Xms20M -Xmx20M -Xmn10M -XX:+PrintGCDetails
 * -XX:+UseSerialGC -XX:SurvivorRatio=8 -XX:MaxTenuringThreshold=1
 */
public class TestTenuringThreshold {
  private static final int _1M = 1024 * 1024;

  public static void testTenuringThreshold() {
    byte[] a1, a2, a3;
    a1 = new byte[_1M / 4];
    // 什么时候进入老年代取决于XX:MaxTenuringThreshold设置
    a2 = new byte[4 * _1M];
    a3 = new byte[4 * _1M];
    a3 = null;
    a3 = new byte[4 * _1M];
  }

  public static void main(String[] args) { testTenuringThreshold(); }
}
