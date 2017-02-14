public class TestThread{
	public static void main(String[] args)throws InterruptedException{
		Thread[] threads = new Thread[8];

		// 创建8个未运行的线程对象
		for(int i = 0;i < threads.length;i++){
			threads[i] = new Thread(new Runnable(){ // 匿名内部类
				public void run(){
					System.out.println(Thread.currentThread().getName());
				}		
			});
		}
		
		// 调用线程对象的start方法使得线程可被调度运行,JVM调用线程对象的run方法
		for(int i = 0;i < threads.length;i++){
			threads[i].start();
		}

		// join 方法阻塞调用线程,直到线程对象退出(包括正常退出和异常退出)
		for(int i = 0;i < threads.length;i++){
			threads[i].join();
		}
		
	}
}
