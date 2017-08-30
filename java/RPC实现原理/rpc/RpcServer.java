package rpc;

import java.io.IOException;

public class RpcServer {
    public static void main(String[] args) {
        try {
            HelloService service = new HelloServiceImpl();
            RpcFramework.export(service,8989);
        } catch(IOException e) {
            e.printStackTrace();
        }
    }

    private static class HelloServiceImpl implements HelloService {
        @Override
        public String sayHello(String msg) {
            String result = "hello world " + msg;
            System.out.println(result);
            return result;
        }
    }
}
