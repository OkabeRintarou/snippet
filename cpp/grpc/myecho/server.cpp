#include "echo.pb.h"
#include "myrpc.h"

namespace example {
class EchoServerImpl : public example::EchoService {
  public:
    virtual void Echo(::google::protobuf::RpcController* controller,
                      const ::example::EchoRequest* request,
                      ::example::EchoResponse* response,
                      ::google::protobuf::Closure* done) override;
  };


  void EchoServerImpl::Echo(::google::protobuf::RpcController *controller, const ::example::EchoRequest *request,
                            ::example::EchoResponse *response, ::google::protobuf::Closure *done) {
    response->set_message(request->message());
    done->Run();
  }
}

int main() {
  MyServer server;
  example::EchoServerImpl echo_service;
  server.add(&echo_service);
  server.start("127.0.0.1", 8086);
  return 0;
}