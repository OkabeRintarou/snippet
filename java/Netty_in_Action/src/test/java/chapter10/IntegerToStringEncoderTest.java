package chapter10;

import io.netty.channel.embedded.EmbeddedChannel;
import junit.framework.TestCase;
import org.junit.Test;

public class IntegerToStringEncoderTest extends TestCase {
    @Test
    public void testEncoder() {
        EmbeddedChannel channel = new EmbeddedChannel(
                new IntegerToStringEncoder()
        );
        for (int i = 0; i < 100; i++) {
            assertTrue(channel.writeOutbound(i));
        }
        assertTrue(channel.finish());

        for (int i = 0; i < 100; i++) {
            String read = channel.readOutbound();
            assertEquals(String.valueOf(i), read);
        }
        assertNull(channel.readOutbound());
    }
}
