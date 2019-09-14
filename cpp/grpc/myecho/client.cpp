#include "myrpc.h"
#include "echo.pb.h"

int main() {
  MyChannel channel;
  channel.init("127.0.0.1", 8086);
  example::EchoService_Stub stub(&channel);
  MyController controller;

  example::EchoRequest request;
  example::EchoResponse response;
  request.set_message("hello, world!");
  stub.Echo(&controller, &request, &response, NULL);
  std::cout << "resp: " << response.message() << std::endl;
  return 0;
}