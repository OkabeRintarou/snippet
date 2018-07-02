import sun.jvm.hotspot.oops.*;
import sun.jvm.hotspot.runtime.VM;
import sun.jvm.hotspot.utilities.U1Array;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;

public class HeapConstantPoolVisitor implements HeapVisitor {
    private static final PrintStream tty = new PrintStream(System.out);
    @Override
    public void prologue(long l) {

    }

    public void doObj(Array oop) {

    }

    public void doOop(Instance oop) {
        InstanceKlass ik = (InstanceKlass)oop.getKlass();
        if (ik.getName().asString().equals("ConstantPoolTestCase")) {
            ConstantPool cp = ik.getConstants();
            Map<String,Short> utf8ToIndex = new HashMap();
            U1Array tags = cp.getTags();
            int len = cp.getLength();
            int ci; // constant pool index
            // collect all modified UTF-8 Strings from Constant Pool
            for (ci = 1; ci < len; ci++) {
                byte cpConstType = tags.at(ci);
                if (cpConstType == ConstantPool.JVM_CONSTANT_Utf8) {
                    Symbol symbol = cp.getSymbolAt(ci);
                    utf8ToIndex.put(symbol.asString(),(short)ci);
                } else if (cpConstType == ConstantPool.JVM_CONSTANT_Long ||
                        cpConstType == ConstantPool.JVM_CONSTANT_Double) {
                    ci++;
                }
            }

            for (ci = 1; ci < len; ci++) {
                int cpConstType = (int)tags.at(ci);
                switch (cpConstType) {
                    case ConstantPool.JVM_CONSTANT_Utf8:
                        tty.print("类型: " + cpConstType);
                        Symbol symbol = cp.getSymbolAt(ci);
                        tty.print(" ;长度: " + (short)symbol.getLength());
                        tty.println(" ;CP[" + ci + "] = modified UTF-8 " + symbol.asString());
                }
            }
        }

    }

    @Override
    public boolean doObj(Oop oop) {
        if (oop instanceof Instance) {
            doOop((Instance)oop);
        } else {
            doObj((Array)oop);
        }

        return false;
    }

    @Override
    public void epilogue() {

    }

    public static void main(String[] args) {
        ObjectHeap heap = VM.getVM().getObjectHeap();
        try {
            heap.iterate(new HeapConstantPoolVisitor());
        } catch (RuntimeException e) {
            e.printStackTrace();
        }
    }
}
