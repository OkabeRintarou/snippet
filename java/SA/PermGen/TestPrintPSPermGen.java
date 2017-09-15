import sun.jvm.hotspot.gc_implementation.parallelScavenge.PSPermGen;  
import sun.jvm.hotspot.gc_implementation.parallelScavenge.ParallelScavengeHeap;  
import sun.jvm.hotspot.gc_implementation.shared.MutableSpace;  
import sun.jvm.hotspot.gc_interface.CollectedHeap;  
import sun.jvm.hotspot.memory.Universe;  
import sun.jvm.hotspot.oops.HeapPrinter;  
import sun.jvm.hotspot.oops.HeapVisitor;  
import sun.jvm.hotspot.oops.ObjectHeap;  
import sun.jvm.hotspot.runtime.VM;  
import sun.jvm.hotspot.tools.Tool;

public class TestPrintPSPermGen extends Tool {
	public static void main(String[] args) {
		TestPrintPSPermGen test = new TestPrintPSPermGen();
		test.start(args);
		test.stop();
	}

	@Override
	public void run() {
		VM vm = VM.getVM();
		Universe universe = vm.getUniverse();
		CollectedHeap heap = universe.heap();
	
		puts("GC heap name: " + heap.kind());		
		if(heap instanceof ParallelScavengeHeap) {
			ParallelScavengeHeap psHeap = (ParallelScavengeHeap)heap;
			PSPermGen perm = psHeap.permGen();
			MutableSpace permObjSpace = perm.objectSpace();
			puts("Perm gen: [" + permObjSpace.bottom() + "," + permObjSpace.end() + ")");
			long permSize = 0;
			for(VM.Flag f : VM.getVM().getCommandLineFlags()) {
				if("PermSize".equals(f.getName())) {
					permSize = Long.parseLong(f.getValue());
					break;
				}
			}
			puts("PermSize: " + permSize);
			
			puts();
			
			ObjectHeap objHeap = vm.getObjectHeap();
			HeapVisitor heapVisitor = new HeapPrinter(System.out);
			objHeap.iteratePerm(heapVisitor);
		}
	}

	public static void puts() {
		System.out.println();
	}

	public static void puts(String s) {
		System.out.println(s);
	}
}
