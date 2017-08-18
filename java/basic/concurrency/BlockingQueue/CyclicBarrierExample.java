import java.util.concurrent.*;

public class CyclicBarrierExample {
	public static void main(String[] args) throws Exception {
		Runnable barrier1Action = new Runnable() {
			public void run() {
				System.out.println("BarrierAction 1 executed");
			}
		};

		Runnable barrier2Action = new Runnable() {
			public void run() {
				System.out.println("BarrierAction 2 executed");
			}
		};

		CyclicBarrier barrier1 = new CyclicBarrier(2,barrier1Action);
		CyclicBarrier barrier2 = new CyclicBarrier(2,barrier2Action);
	
		CyclicBarrierRunnable runnable1 = new CyclicBarrierRunnable(barrier1,barrier2);
		CyclicBarrierRunnable runnable2 = new CyclicBarrierRunnable(barrier1,barrier2);
		Thread thread1 = new Thread(runnable1);
		Thread thread2 = new Thread(runnable2);
		thread1.start();thread2.start();
		thread1.join();thread2.join();
	}
}

class CyclicBarrierRunnable implements Runnable {
	CyclicBarrier barrier1 = null;
	CyclicBarrier barrier2 = null;
	
	public CyclicBarrierRunnable(CyclicBarrier barrier1,CyclicBarrier barrier2) {
		this.barrier1 = barrier1;
		this.barrier2 = barrier2;
	}

	public void run() {
		try {
			Thread.sleep(1000);
			System.out.println(Thread.currentThread().getName() + " waiting at barrier1");
			this.barrier1.await();
		
			Thread.sleep(1000);
			System.out.println(Thread.currentThread().getName() + " waiting at barrier2");
			this.barrier2.await();

			System.out.println(Thread.currentThread().getName() + " done!");
		} catch(InterruptedException e) {
			e.printStackTrace();
		} catch(BrokenBarrierException e) {
			e.printStackTrace();
		}
	}
}
