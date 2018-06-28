package chapter10;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.embedded.EmbeddedChannel;
import junit.framework.TestCase;
import org.junit.Test;

public class IntegerToStringDecoderTest extends TestCase {
    @Test
    public void testDecoder() {
        ByteBuf buf = Unpooled.buffer();

        for (int i = 0; i < 10; i++) {
            buf.writeInt(i);
        }

        EmbeddedChannel channel = new EmbeddedChannel(
                new IntegerToStringDecoder()
        );

        assertTrue(channel.writeInbound((int) '1'));
        for (int i = 0; i < 10; i++) {
            assertTrue(channel.writeInbound(buf.readInt()));
        }

        assertTrue(channel.finish());

        assertEquals("49", channel.readInbound());
        for (int i = 0; i < 10; i++) {
            String read = channel.readInbound();
            assertEquals(String.valueOf(i), read);
        }

        assertNull(channel.readInbound());

    }
}
