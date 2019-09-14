#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>
#include <boost/asio.hpp>
#include <google/protobuf/service.h>

class MyChannel : public ::google::protobuf::RpcChannel {
public:
  void init(const std::string &ip, const int port);

  virtual void CallMethod(const ::google::protobuf::MethodDescriptor *method,
                          ::google::protobuf::RpcController *ctn,
                          const ::google::protobuf::Message *request,
                          ::google::protobuf::Message *response,
                          ::google::protobuf::Closure *) override ;

private:
  std::shared_ptr<boost::asio::io_service> _io;
  std::shared_ptr<boost::asio::ip::tcp::socket> _socket;
};

class MyController : public ::google::protobuf::RpcController {
public:
  MyController();

  virtual ~MyController();

  void Reset() override;

  bool Failed() const override;

  std::string ErrorText() const override;

  void StartCancel() override;

  void SetFailed(const std::string &reason) override;

  bool IsCanceled() const override;

  void NotifyOnCancel(google::protobuf::Closure *callback) override;
};

class MyServer {
public:
  void add(::google::protobuf::Service *service);

  void start(const std::string &ip, const int port);


private:
  void dispatch_msg(const std::string &service_name,
                    const std::string &method_name,
                    const std::string &serialized_data,
                    const std::shared_ptr<boost::asio::ip::tcp::socket> &sock);

  void on_resp_msg_filled(const std::shared_ptr<boost::asio::ip::tcp::socket> sock,
                          std::pair<::google::protobuf::Message *, ::google::protobuf::Message *> msgs);

  static void pack_message(const ::google::protobuf::Message *msg, std::string *serialized_data);

private:
  struct ServiceInfo {
    ::google::protobuf::Service *service;
    const ::google::protobuf::ServiceDescriptor *descriptor;
    std::unordered_map<std::string, const ::google::protobuf::MethodDescriptor *> mds;
  };
  std::unordered_map<std::string, ServiceInfo> _services_map;
};