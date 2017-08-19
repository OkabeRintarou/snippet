import java.util.concurrent.*;

public class ExecutorServiceExample {
	public static void main(String[] args) throws InterruptedException,ExecutionException {
		ExecutorService es = Executors.newSingleThreadExecutor();
		es.execute(new Runnable() {
			public void run() {
				System.out.println("Asynchronous task");
			}		
		});
		
		Future future = es.submit(new Runnable() {
			public void run() {
				System.out.println("Asynchronous task");
			}		
		});

		System.out.println("future.get() = " + future.get());
		
		future = es.submit(new Callable<String>() {
			public String call() throws Exception {
				System.out.println("Asynchrounous Callable");
				return "Callable Result";
			}
		});
		System.out.println("future.get() = " + future.get());
		
		es.shutdown();
	}
}
