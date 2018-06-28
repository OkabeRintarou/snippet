package chapter10;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageDecoder;

import java.util.List;

public class IntegerToStringDecoder extends MessageToMessageDecoder<Integer> {

    @Override
    protected void decode(ChannelHandlerContext channelHandlerContext,
                          Integer in, List<Object> out) throws Exception {
        out.add(String.valueOf(in));
    }
}
