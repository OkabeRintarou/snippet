import sun.jvm.hotspot.oops.*;
import sun.jvm.hotspot.runtime.VM;
import sun.jvm.hotspot.tools.Tool;

public class KlassKicker extends Tool {

	public static void main(String[] args) throws Exception {
		KlassKicker kk = new KlassKicker();
		kk.start(args);
		kk.stop();
	}

	@Override
	public void run() {
		VM vm = VM.getVM();
		ObjectHeap heap = vm.getObjectHeap();
		heap.iterate(new HeapVisitor() {
			@Override
			public void prologue(long l) {
				
			}	

			@Override
			public boolean doObj(Oop oop) {
				System.out.println("///////////////////////////////////////");
				System.out.println("OOP#"+oop);
				oop.iterate(new OopPrinter(System.out),true);
				System.out.println("///////////////////////////////////////");
				System.out.println("OOP.KLASS#" + oop.getKlass());
				oop.getKlass().iterate(new OopPrinter(System.out),true);
				System.out.println("///////////////////////////////////////");
				System.out.println("OOP.KLASS.MIRROR#" + oop.getKlass().getJavaMirror());
				oop.getKlass().getJavaMirror().iterate(new OopPrinter(System.out),true);
				System.out.println("///////////////////////////////////////");
				
				System.out.println("///////////////////////////////////////");
				System.out.println("OOP.KLASS.KLASS#" + oop.getKlass().getKlass());
				oop.getKlass().getKlass().iterate(new OopPrinter(System.out),true);
				System.out.println("///////////////////////////////////////");
				System.out.println("OOP.KLASS.KLASS.KLASS#" + oop.getKlass().getKlass().getKlass());
				oop.getKlass().getKlass().getKlass().iterate(new OopPrinter(System.out),true);
				System.out.println("///////////////////////////////////////");
				System.out.println("OOP.KLASS.KLASS.KLASS.KLASS#" + oop.getKlass().getKlass().getKlass().getKlass());
				oop.getKlass().getKlass().getKlass().getKlass().iterate(new OopPrinter(System.out),true);
				return false;
			}		

			@Override
			public void epilogue() {

			}
		},new ObjectHeap.ObjectFilter() {
			@Override
			public boolean canInclude(Oop oop) {
				Klass klass = oop.getKlass();
				return klass.getName() != null && "Foo".equals(klass.getName().asString());
			}
		});
	}
}
