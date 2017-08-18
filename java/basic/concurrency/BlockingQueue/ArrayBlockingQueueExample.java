import java.util.concurrent.*;

public class ArrayBlockingQueueExample {
	public static void main(String[] args) throws Exception {
		BlockingQueue<Integer> queue = new ArrayBlockingQueue<Integer>(1024);
		Consumer consumer = new Consumer(queue);
		Producer producer = new Producer(queue);
		Thread thread1 = new Thread(consumer);
		Thread thread2 = new Thread(producer);
		thread1.start();
		thread2.start();
		thread1.join();
		thread2.join();
	}
}

class Producer implements Runnable {
	private BlockingQueue queue;

	public Producer(BlockingQueue q) {
		queue = q;
	}

	public void run() {
		int number = 0;
		for(;;) {
			System.out.println("Producer produce number " + number);
			try {
				queue.put(number++);
			} catch(InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
}

class Consumer implements Runnable {
	private BlockingQueue queue;

	public Consumer(BlockingQueue q) {
		queue = q;
	}

	public void run() {
		try {
			for(;;) {
				int number = (Integer)queue.take();
				System.out.println("Consumber consume number " + number);
			}
		} catch(InterruptedException e) {
			e.printStackTrace();
		}
	}
}
