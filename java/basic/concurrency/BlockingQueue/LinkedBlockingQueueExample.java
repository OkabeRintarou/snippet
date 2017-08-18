import java.util.concurrent.*;

public class LinkedBlockingQueueExample {
	public static void main(String[] args) throws Exception {
		BlockingQueue<String> bounded = new LinkedBlockingQueue<String>(1024);
		BlockingQueue<String> unbounded = new LinkedBlockingQueue<String>();
		
		bounded.put("1");
		System.out.println(bounded.take());
	}
}
