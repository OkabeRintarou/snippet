import java.net.Socket;
import java.net.ServerSocket;

public class Server{
	public static void main(String[] args)throws Exception{
		ServerSocket server = new ServerSocket(20006);
		Socket client = null;
		for(;;){
			client = server.accept();
			new Thread(new ServerThread(client)).start();
		}
	}
}
