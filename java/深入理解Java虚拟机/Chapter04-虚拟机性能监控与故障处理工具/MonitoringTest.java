import java.util.List;
import java.util.ArrayList;

public class MonitoringTest {
	static class OOMObject {
		public byte[] placeholder = new byte[64 * 1024];
	}

	public static void fillHeap(int num) throws Exception {

		List<OOMObject> list = new ArrayList<OOMObject>();

		for (int i = 0; i < num; i++) {
			Thread.sleep(50);
			list.add(new OOMObject());
		}
		System.gc();
	}

	public static void main(String[] args) throws Exception {
		fillHeap(1000);
	}
}
