import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.BufferedReader;
import java.net.Socket;
import java.net.SocketTimeoutException;

public class Client{
	public static void main(String[] args)throws IOException{
		Socket client = new Socket("127.0.0.1",20006);
		client.setSoTimeout(10000);
		BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
		PrintStream out = new PrintStream(client.getOutputStream());
		BufferedReader buf = new BufferedReader(new InputStreamReader(client.getInputStream()));
		for(;;){
			System.out.print(">>> ");
			String str = input.readLine();
			out.println(str);
			if("bye".equals(str)){
				break;
			}else{
				try{
					String echo = buf.readLine();
					System.out.println(echo);
				}catch(SocketTimeoutException e){
					System.out.println("Timeout,no response");
				}
			}
		}
	}
}
