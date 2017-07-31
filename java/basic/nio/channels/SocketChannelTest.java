import java.nio.channels.SocketChannel;
import java.nio.ByteBuffer;
import java.net.*;

public class SocketChannelTest{
	public static void main(String[] args)throws Exception{
		SocketChannel socketChannel = SocketChannel.open();
		socketChannel.connect(new InetSocketAddress("qq.com",80));
		
		String request = "GET HTTP/1.0 /\r\n\r\n";
		ByteBuffer buf = ByteBuffer.allocate(1024);
		buf.clear();
		buf.put(request.getBytes());
		buf.flip();
		while(buf.hasRemaining()){
			socketChannel.write(buf);
		}
		buf.clear();
		socketChannel.read(buf);
		buf.flip();
		while(buf.hasRemaining()){
			System.out.print((char)buf.get());
		}
	}
}
