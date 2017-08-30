package rpc;

public class RpcClient {
    public static void main(String[] args) {
        try {
            // 引用服务
            HelloService service = RpcFramework.refer(HelloService.class,"127.0.0.1",8989);
            for(int i = 0;i < Integer.MAX_VALUE;i++) {
                String hello = service.sayHello("rpc " + i);
                System.out.println(hello);
                Thread.sleep(1000);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
