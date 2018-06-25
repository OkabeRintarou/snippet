package chapter01;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

public class BlockingServer {
    public static void main(String[] args) {
        try {
            ServerSocket serverSocket = new ServerSocket(6666);
            for (; ; ) {
                Socket clientSocket = serverSocket.accept();
                System.out.println("Connection from " + clientSocket);
                BufferedReader in = new BufferedReader(
                        new InputStreamReader(clientSocket.getInputStream()));
                PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
                String request, response;
                while ((request = in.readLine()) != null) {
                    if ("Done".equals(request)) {
                        break;
                    }
                    response = "Hello," + request;
                    out.println(response);
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
