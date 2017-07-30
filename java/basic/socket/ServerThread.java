import java.io.BufferedReader;
import java.io.PrintStream;
import java.io.InputStreamReader;
import java.net.Socket;

public class ServerThread implements Runnable{
	private Socket client = null;
	
	public ServerThread(Socket c){
		client = c;
	}

	@Override
	public void run(){
		try{
			PrintStream out = new PrintStream(client.getOutputStream());
			BufferedReader buf = new BufferedReader(new InputStreamReader(client.getInputStream()));
			for(;;){
				String str = buf.readLine();
				if(str == null || str.isEmpty() || "bye".equals(str)){
					break;
				}else{
					out.println("echo:" + str);
				}

			}
			out.close();
			client.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
}
