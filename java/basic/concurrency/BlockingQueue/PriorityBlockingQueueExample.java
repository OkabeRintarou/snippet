import java.util.concurrent.*;

public class PriorityBlockingQueueExample {
	public static void main(String[] args) throws Exception {
		BlockingQueue<String> queue = new PriorityBlockingQueue<String>();
		queue.put("world");
		queue.put("hello");
		System.out.println(queue.take());
		System.out.println(queue.take());
	}
}
