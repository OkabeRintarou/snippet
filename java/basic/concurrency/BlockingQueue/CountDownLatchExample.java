import java.util.concurrent.*;

public class CountDownLatchExample {
	public static void main(String[] args) throws Exception {
		CountDownLatch latch = new CountDownLatch(3);
		new Thread(new Waiter(latch)).start();
		new Thread(new Decrementer(latch)).start();
		Thread.sleep(5000);
	}
}

class Waiter implements Runnable {

	CountDownLatch latch = null;

	public Waiter(CountDownLatch latch) {
		this.latch = latch;
	}

	public void run() {
		try {
			latch.await();
		} catch(InterruptedException e) {
			e.printStackTrace();
		}

		System.out.println("Waiter Released");
	}
}

class Decrementer implements Runnable {

	CountDownLatch latch = null;

	public Decrementer(CountDownLatch latch) {
		this.latch = latch;
	}

	public void run() {
		try {
			Thread.sleep(1000);
			latch.countDown();
			Thread.sleep(1000);
			latch.countDown();
			Thread.sleep(1000);
			latch.countDown();
		} catch(InterruptedException e) {
			e.printStackTrace();
		}
	}
}
