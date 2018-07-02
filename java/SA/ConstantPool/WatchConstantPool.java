import sun.jvm.hotspot.oops.ObjectHeap;
import sun.jvm.hotspot.runtime.VM;
import sun.jvm.hotspot.tools.Tool;

import java.io.IOException;

public class WatchConstantPool extends Tool {
    @Override
    public void run() {
        try {
            watchConstantPool();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void watchConstantPool() throws IOException {
        ObjectHeap heap = VM.getVM().getObjectHeap();
        heap.iterate(new HeapConstantPoolVisitor());
    }

    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.printf("usage: %s <pid>",WatchConstantPool.class.getSimpleName());
            System.exit(0);
        }

        WatchConstantPool watcher = new WatchConstantPool();
        watcher.execute(args);
        watcher.stop();
    }
}
