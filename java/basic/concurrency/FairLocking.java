import java.util.concurrent.locks.*;

public class FairLocking {
	private static final boolean FAIR = false;
	private static final int NUM_THREADS = 3;

	private static volatile int expectedIndex = 0;

	public static void main(String[] args) throws Exception {
		ReentrantReadWriteLock.WriteLock lock = new ReentrantReadWriteLock(FAIR).writeLock();

		lock.lock();

		for (int i = 0; i < NUM_THREADS; i++) {
			new Thread(new ExampleRunnable(i,lock)).start();
			Thread.sleep(10);
		}

		lock.unlock();
	}

	private static class ExampleRunnable implements Runnable {
		
		private final int index;
		private final ReentrantReadWriteLock.WriteLock writeLock;

		public ExampleRunnable(int i, ReentrantReadWriteLock.WriteLock l) {
			index = i;
			writeLock = l;
		}

		public void run() {
			for (;;) {
				writeLock.lock();
				System.out.printf("[%d] own the lock\n",index);
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
				
				}

				if (index != expectedIndex) {
					System.out.printf("Unexpected thread obtained lock! " +
						"Expected: %d Actual: %d\n",expectedIndex,index);
					System.exit(0);
				}
				expectedIndex = (expectedIndex + 1) % NUM_THREADS;
				writeLock.unlock();
			}
		}
	}
}

