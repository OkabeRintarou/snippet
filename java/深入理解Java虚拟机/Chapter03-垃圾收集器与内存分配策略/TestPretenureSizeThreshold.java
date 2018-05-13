/**
 * VM Args: -verbose:gc -client -Xms20M -Xmx20M -Xmn10M -XX:+PrintGCDetails
 * -XX:+UseSerialGC -XX:SurvivorRatio=8 -XX:PretenureSizeThreshold=3145728
 */
public class TestPretenureSizeThreshold {
  private static final int _1M = 1024 * 1024;

  public static void testPretenureSizeThreshold() {
    byte[] a;
    a = new byte[4 * _1M]; // 直接分配在老年代中
  }

  public static void main(String[] args) { testPretenureSizeThreshold(); }
}
