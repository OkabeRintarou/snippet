import java.util.concurrent.*;

public class BlockingQueueExample {
	public static void main(String[] args) throws Exception {
		BlockingQueue queue = new ArrayBlockingQueue(1024);
		
		Producer producer = new Producer(queue);
		Consumer consumer = new Consumer(queue);
		
		new Thread(producer).start();
		new Thread(consumer).start();
		Thread.sleep(4000);
	}
}

class Producer implements Runnable {
	protected BlockingQueue queue = null;

	public Producer(BlockingQueue q) {
		queue = q;
	}

	public void run() {
		try {
			queue.put("1");
			Thread.sleep(1000);
			queue.put("2");
			Thread.sleep(1000);
			queue.put("3");
		} catch(InterruptedException e) {
			e.printStackTrace();
		}
	}
}

class Consumer implements Runnable {
	protected BlockingQueue queue;

	public Consumer(BlockingQueue q) {
		queue = q;
	}

	public void run() {
		try {
			System.out.println(queue.take());
			System.out.println(queue.take());
			System.out.println(queue.take());
		} catch(InterruptedException e) {
			e.printStackTrace();
		}
	}
}
