all:

TestMinorGC:TestMinorGC.class
	java -client -verbose:gc -Xms20M -Xmx20M -Xmn10M -XX:+PrintGCDetails -XX:+UseSerialGC -XX:SurvivorRatio=8  $@
TestPretenureSizeThreshold:TestPretenureSizeThreshold.class
	java -verbose:gc -client -Xms20M -Xmx20M -Xmn10M -XX:+PrintGCDetails -XX:+UseSerialGC -XX:SurvivorRatio=8 -XX:PretenureSizeThreshold=3145728 $@
TestTenuringThreshold1:TestTenuringThreshold.class
	java -verbose:gc -client -Xms20M -Xmx20M -Xmn10M -XX:+PrintGCDetails -XX:+UseSerialGC -XX:SurvivorRatio=8 -XX:MaxTenuringThreshold=1 TestTenuringThreshold
TestTenuringThreshold15:TestTenuringThreshold.class
	java -verbose:gc -client -Xms20M -Xmx20M -Xmn10M -XX:+PrintGCDetails -XX:+UseSerialGC -XX:SurvivorRatio=8 -XX:MaxTenuringThreshold=15 TestTenuringThreshold
TestTenuringThreshold2:TestTenuringThreshold2.class
	java -verbose:gc -client -Xms20M -Xmx20M -Xmn10M -XX:+PrintGCDetails -XX:+UseSerialGC -XX:SurvivorRatio=8 -XX:MaxTenuringThreshold=15 $@
