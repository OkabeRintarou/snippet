package chapter10;

import io.netty.buffer.ByteBuf;
import io.netty.channel.embedded.EmbeddedChannel;
import junit.framework.TestCase;
import org.junit.Test;

public class ShortToByteEncoderTest extends TestCase {
    @Test
    public void testEncoder() {
        short[] data = new short[]{0x1234, 0x4567, 0x4765, 0x0110};
        EmbeddedChannel channel = new EmbeddedChannel(
                new ShortToByteEncoder()
        );

        for (short msg : data) {
            assertTrue(channel.writeOutbound(msg));
        }

        assertTrue(channel.finish());

        for (int i = 0; i < data.length; i++) {
            ByteBuf read = channel.readOutbound();

            short byte1 = read.readByte();
            short byte2 = read.readByte();

            short s1 = (short) ((byte1 << 8) | byte2);
            short s2 = (short) ((byte2 << 8) | byte1);
            assertTrue(s1 == data[i] || s2 == data[i]);
        }

    }
}
