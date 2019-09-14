#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <boost/asio.hpp>
#include <google/protobuf/service.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/stubs/common.h>
#include "myrpc.h"
#include "rpc_meta.pb.h"

void MyServer::add(::google::protobuf::Service *service) {
  ServiceInfo service_info;
  service_info.service = service;
  service_info.descriptor = service->GetDescriptor();
  for (int i = 0; i < service_info.descriptor->method_count(); ++i) {
    const ::google::protobuf::MethodDescriptor *md = service_info.descriptor->method(i);
    service_info.mds[md->name()] = md;
  }
  _services_map[service_info.descriptor->name()] = service_info;
}

void MyServer::start(const std::string &ip, const int port) {
  boost::asio::io_service io;
  boost::asio::ip::tcp::acceptor acceptor(
    io, boost::asio::ip::tcp::endpoint(boost::asio::ip::address::from_string(ip), port)
  );

  for (;;) {
    auto sock = std::make_shared<boost::asio::ip::tcp::socket>(io);
    acceptor.accept(*sock);

    std::cout << "recv from client: " << sock->remote_endpoint().address() << std::endl;

    // recv length of meta data
    char meta_size[sizeof(int)];
    sock->receive(boost::asio::buffer(meta_size));

    int meta_length = *(int *) (meta_size);

    // recv meta data
    std::vector<char> meta_data(meta_length, 0);
    sock->receive(boost::asio::buffer(meta_data));

    example::RpcMeta meta;
    meta.ParseFromString(std::string(&meta_data[0], meta_data.size()));

    std::vector<char> data(meta.data_size(), 0);
    sock->receive(boost::asio::buffer(data));

    dispatch_msg(meta.service_name(), meta.method_name(), std::string(&data[0], data.size()), sock);
  }
}

void MyServer::dispatch_msg(const std::string &service_name,
                            const std::string &method_name,
                            const std::string &serialized_data,
                            const std::shared_ptr<boost::asio::ip::tcp::socket> &sock) {
  auto &service_info = _services_map[service_name];
  auto service = service_info.service;
  auto md = service_info.mds[method_name];

  std::cout << "recv service name: " << service_name << std::endl;
  std::cout << "recv method name: " << method_name << std::endl;
  std::cout << "recv type: " << md->input_type()->name() << std::endl;
  std::cout << "resp type: " << md->output_type()->name() << std::endl;

  auto recv_message = service->GetRequestPrototype(md).New();
  recv_message->ParseFromString(serialized_data);
  auto resp_message = service->GetResponsePrototype(md).New();

  MyController controller;
  auto done = ::google::protobuf::NewCallback(
    this,
    &MyServer::on_resp_msg_filled,
    sock,
    std::make_pair(recv_message, resp_message));

  service->CallMethod(md, &controller, recv_message, resp_message, done);
}

void MyServer::on_resp_msg_filled(const std::shared_ptr<boost::asio::ip::tcp::socket> sock,
                                  std::pair<::google::protobuf::Message *, ::google::protobuf::Message *> msgs) {

  std::string resp_str;
  pack_message(msgs.second, &resp_str);
  sock->send(boost::asio::buffer(resp_str));
}

void MyServer::pack_message(const ::google::protobuf::Message *msg, std::string *serialized_data) {
  int serialized_size = msg->ByteSize();
  serialized_data->assign((const char *) &serialized_size, sizeof(serialized_size));
  msg->AppendToString(serialized_data);
}

bool MyController::Failed() const {
  return false;
}

std::string MyController::ErrorText() const {
  return std::string();
}

void MyController::StartCancel() {

}

void MyController::SetFailed(const std::string &reason) {

}

bool MyController::IsCanceled() const {
  return false;
}

void MyController::NotifyOnCancel(google::protobuf::Closure *callback) {

}

MyController::MyController() {

}

MyController::~MyController() {

}

void MyController::Reset() {

}

void MyChannel::init(const std::string &ip, const int port) {
  _io = std::make_shared<boost::asio::io_service>();
  _socket = std::make_shared<boost::asio::ip::tcp::socket>(*_io);
  boost::asio::ip::tcp::endpoint ep(boost::asio::ip::address::from_string(ip), port);
  _socket->connect(ep);
}

void MyChannel::CallMethod(const ::google::protobuf::MethodDescriptor *method, ::google::protobuf::RpcController *ctn,
                           const ::google::protobuf::Message *request, ::google::protobuf::Message *response,
                           ::google::protobuf::Closure *) {
  std::string serialized_data = request->SerializeAsString();

  example::RpcMeta meta;
  meta.set_service_name(method->service()->name());
  meta.set_method_name(method->name());
  meta.set_data_size(serialized_data.size());

  std::string serialized_str = meta.SerializeAsString();

  int serialized_size = serialized_str.size();
  serialized_str.insert(0, (const char *) &serialized_size, sizeof(int));
  serialized_str.append(serialized_data);

  _socket->send(boost::asio::buffer(serialized_str));

  char resp_data_size[sizeof(int)];
  _socket->receive(boost::asio::buffer(resp_data_size));

  int resp_data_len = *(int *) resp_data_size;
  std::vector<char> resp_data(resp_data_len, 0);
  _socket->receive(boost::asio::buffer(resp_data));

  response->ParseFromString(std::string(&resp_data[0], resp_data.size()));
}
