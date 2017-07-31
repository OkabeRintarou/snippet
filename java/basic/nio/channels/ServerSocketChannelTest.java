import java.nio.channels.*;
import java.nio.*;
import java.net.*;

public class ServerSocketChannelTest{
	public static void main(String[] args)throws Exception{
		ServerSocketChannel serverSocketChannel = ServerSocketChannel.open();
		serverSocketChannel.bind(new InetSocketAddress(9999));
		serverSocketChannel.configureBlocking(false);
		for(;;){
			SocketChannel clientChannel = serverSocketChannel.accept();
			if(clientChannel == null){
				System.out.println("<no connection>");
			}
		}
	}
}
