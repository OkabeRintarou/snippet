package chapter05;

import io.netty.buffer.*;
import io.netty.util.IllegalReferenceCountException;

import java.nio.charset.Charset;

public class TestByteBuf {

    public static void testDuplicate() {
        ByteBuf buf = Unpooled.wrappedBuffer("Hello,World!".getBytes());
        ByteBuf dup = buf.duplicate();
        dup.setByte(0, 'h');
        assert dup.getByte(0) == buf.getByte(0);
    }

    public static void testSlice() {
        Charset utf8 = Charset.forName("UTF-8");
        ByteBuf buf = Unpooled.copiedBuffer("Netty in Action rocks!", utf8);
        ByteBuf slice = buf.slice(0, 15);
        System.out.println(slice.toString(utf8));
        buf.setByte(0, (byte) 'J');
        assert buf.getByte(0) == slice.getByte(0);
    }

    public static void testCopy() {
        Charset utf8 = Charset.forName("UTF-8");
        ByteBuf buf = Unpooled.copiedBuffer("Netty in Action rocks!", utf8);
        ByteBuf copy = buf.copy(0, 15);
        buf.setByte(0, (byte) 'J');
        assert buf.getByte(0) != copy.getByte(0);
    }

    public static void testGetSet() {
        Charset utf8 = Charset.forName("UTF-8");
        ByteBuf buf = Unpooled.copiedBuffer("Netty in Action rocks!", utf8);
        System.out.println((char) buf.getByte(0));
        int readerIndex = buf.readerIndex();
        int writerIndex = buf.writerIndex();
        buf.setByte(0, (byte) 'J');
        System.out.println((char) buf.getByte(0));
        assert readerIndex == buf.readerIndex();
        assert writerIndex == buf.writerIndex();
    }

    public static void testRead() {
        Charset utf8 = Charset.forName("UTF-8");
        ByteBuf buf = Unpooled.copiedBuffer("Hello", utf8);
        ByteBuf buf2 = Unpooled.copiedBuffer("World", utf8);
        System.out.println(buf2.toString(utf8));
        buf2.clear();
        System.out.println(buf.readerIndex());
        buf.readBytes(buf2, 0, 5);
        System.out.println(buf2.toString(utf8));
        System.out.println(buf.readerIndex());
    }

    public static void byteUnpooled() {
        ByteBuf buf = Unpooled.buffer(10);
        System.out.println("capacity: " + buf.capacity());
        System.out.println("readerIndex: " + buf.readerIndex());
        System.out.println("writerIndex: " + buf.writerIndex());
        System.out.println("maxCapacity: " + buf.maxCapacity());
        for (int i = 0; i < 11; i++) {
            buf.writeByte(i);
        }
        System.out.println("capacity: " + buf.capacity());
        System.out.println("readerIndex: " + buf.readerIndex());
        System.out.println("writerIndex: " + buf.writerIndex());

    }

    public static void testReferenceCount() {
        ByteBuf buf = Unpooled.buffer();
        System.out.println(buf.refCnt());
        boolean released = buf.release();
        System.out.println("released? " + released);

        try {
            buf.getByte(0);
        } catch (IllegalReferenceCountException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        testDuplicate();
        testSlice();
        testCopy();
        testGetSet();
        testRead();
        byteUnpooled();
        testReferenceCount();
    }
}
