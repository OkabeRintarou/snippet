import java.util.Random;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/*
    如何使用Condition对象:
    Condition cond = mutex.newCondition();
    ...
    mutex.lock();
    try{
        while(!property){ // not happy
            cond.await();
        }
        ... // happy:property must hold
    }finally{
        mutex.unlock();
    }

    说明:一个线程要等待某个特性满足该特性.如果不满足,那么线程应该调用await来释放锁,然后休眠直到另一个
    线程唤醒它.当线程被唤醒时无法保证该特性是满足的.await方法有可能出现假返回或者可能出现给条件发出信号
    的线程唤醒了太多的休眠线程.无论是哪种原因,线程都必须再次测试特性,如果发现特性不满足,那么必须再次调用
    await.

    这种将方法、互斥锁和条件对象组合在一起的整体称为管程.
    Java通过synchronized块和方法以及内置的wait()、notify()和notifyAll()方法为管程提供了内置支持
 */
public class LockedQueue<T> {
    final Lock lock = new ReentrantLock();
    final Condition notFull = lock.newCondition();
    final Condition notEmpty = lock.newCondition();
    final T[] items;
    int tail,head,count;

    public LockedQueue(int capacity){
        items = (T[])new Object[capacity];
    }

    public void enq(T x)throws InterruptedException{
        lock.lock();
        try{
            while(count == items.length){
                System.out.println("队列已满," + Thread.currentThread().getName() +
                    "等待元素被消费");
                notFull.await();
            }
            items[tail] = x;
            if(++tail == items.length){
                tail = 0;
            }
            ++count;
            notEmpty.signal();
            /* 把上面这行代码改成如下可能出现唤醒丢失问题
            if(count == 1){
                notEmpty.singal();
            }
            原因:当线程A和B阻塞在notEmpty上时,线程C执行了notEmpty.signal()唤醒了一个线程A,
            但是在A或的锁从await返回之前,另一个线程获得了锁,并又向队列中加入一个元素,此时count等于2
            并不会调用notEmpty.signal(),于是A将获得锁,移走第一个元素,而B却成为唤醒丢失的受害者,此时
            缓冲区中有一个等待消费的元素,B却要永远地等待.
            解决方案:
            (1) 总是通知所有等待条件的线程,而不是仅仅通知一个
            (2) 等待时指定一个超时时限
            */
        }finally {
            lock.unlock();
        }
    }

    public T deq()throws InterruptedException{
        lock.lock();
        try{
            while(count == 0){
                System.out.println("队列已空," + Thread.currentThread().getName() +
                    "等待元素被生产");
                notEmpty.await();
            }
            T x = items[head];
            if(++head == items.length){
                head = 0;
            }
            --count;
            notFull.signal();
            return x;
        }finally {
            lock.unlock();
        }
    }


    public static void main(String[] args)throws InterruptedException {
        LockedQueue<Integer> lockedQueue = new LockedQueue<Integer>(10);

        Thread[] threads = new Thread[5];
        for(int i = 0;i < 3;i++){
            threads[i] = new Producer(lockedQueue);
        }
        for(int i = 3;i < threads.length;i++){
            threads[i] = new Consumer(lockedQueue);
        }

        for(int i = 0;i < threads.length;i++){
            threads[i].start();
        }
        for(int i = 0;i < threads.length;i++){
            threads[i].join();
        }
    }
}

class Producer extends Thread{
    final LockedQueue<Integer> lq;
    final Random rnd = new Random();

    public Producer(LockedQueue<Integer> queue){
        lq = queue;
    }

    @Override
    public void run(){
        int count = 0;
        try{
            for(;;){
                lq.enq(count++);
                System.out.println(Thread.currentThread().getName() + " 将元素" + count + "放入队列");
                Thread.sleep(rnd.nextInt(10) * 100);

            }
        }catch(InterruptedException e){
            e.printStackTrace();
        }
    }
}

class Consumer extends Thread{
    final LockedQueue<Integer> lq;
    final Random rnd = new Random();

    public Consumer(LockedQueue<Integer> queue){
        lq = queue;
    }

    @Override
    public void run(){
        try {
            for(;;){
                int item = lq.deq();
                System.out.println(Thread.currentThread().getName() + "消费了元素" + item);
                Thread.sleep(rnd.nextInt(10) * 100);
            }

        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}