// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: rpc_meta.proto

#ifndef PROTOBUF_rpc_5fmeta_2eproto__INCLUDED
#define PROTOBUF_rpc_5fmeta_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace example {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_rpc_5fmeta_2eproto();
void protobuf_AssignDesc_rpc_5fmeta_2eproto();
void protobuf_ShutdownFile_rpc_5fmeta_2eproto();

class RpcMeta;

// ===================================================================

class RpcMeta : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:example.RpcMeta) */ {
 public:
  RpcMeta();
  virtual ~RpcMeta();

  RpcMeta(const RpcMeta& from);

  inline RpcMeta& operator=(const RpcMeta& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const RpcMeta& default_instance();

  void Swap(RpcMeta* other);

  // implements Message ----------------------------------------------

  inline RpcMeta* New() const { return New(NULL); }

  RpcMeta* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const RpcMeta& from);
  void MergeFrom(const RpcMeta& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(RpcMeta* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string service_name = 1;
  void clear_service_name();
  static const int kServiceNameFieldNumber = 1;
  const ::std::string& service_name() const;
  void set_service_name(const ::std::string& value);
  void set_service_name(const char* value);
  void set_service_name(const char* value, size_t size);
  ::std::string* mutable_service_name();
  ::std::string* release_service_name();
  void set_allocated_service_name(::std::string* service_name);

  // optional string method_name = 2;
  void clear_method_name();
  static const int kMethodNameFieldNumber = 2;
  const ::std::string& method_name() const;
  void set_method_name(const ::std::string& value);
  void set_method_name(const char* value);
  void set_method_name(const char* value, size_t size);
  ::std::string* mutable_method_name();
  ::std::string* release_method_name();
  void set_allocated_method_name(::std::string* method_name);

  // optional int32 data_size = 3;
  void clear_data_size();
  static const int kDataSizeFieldNumber = 3;
  ::google::protobuf::int32 data_size() const;
  void set_data_size(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:example.RpcMeta)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  bool _is_default_instance_;
  ::google::protobuf::internal::ArenaStringPtr service_name_;
  ::google::protobuf::internal::ArenaStringPtr method_name_;
  ::google::protobuf::int32 data_size_;
  mutable int _cached_size_;
  friend void  protobuf_AddDesc_rpc_5fmeta_2eproto();
  friend void protobuf_AssignDesc_rpc_5fmeta_2eproto();
  friend void protobuf_ShutdownFile_rpc_5fmeta_2eproto();

  void InitAsDefaultInstance();
  static RpcMeta* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// RpcMeta

// optional string service_name = 1;
inline void RpcMeta::clear_service_name() {
  service_name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& RpcMeta::service_name() const {
  // @@protoc_insertion_point(field_get:example.RpcMeta.service_name)
  return service_name_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void RpcMeta::set_service_name(const ::std::string& value) {
  
  service_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:example.RpcMeta.service_name)
}
inline void RpcMeta::set_service_name(const char* value) {
  
  service_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:example.RpcMeta.service_name)
}
inline void RpcMeta::set_service_name(const char* value, size_t size) {
  
  service_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:example.RpcMeta.service_name)
}
inline ::std::string* RpcMeta::mutable_service_name() {
  
  // @@protoc_insertion_point(field_mutable:example.RpcMeta.service_name)
  return service_name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* RpcMeta::release_service_name() {
  // @@protoc_insertion_point(field_release:example.RpcMeta.service_name)
  
  return service_name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void RpcMeta::set_allocated_service_name(::std::string* service_name) {
  if (service_name != NULL) {
    
  } else {
    
  }
  service_name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), service_name);
  // @@protoc_insertion_point(field_set_allocated:example.RpcMeta.service_name)
}

// optional string method_name = 2;
inline void RpcMeta::clear_method_name() {
  method_name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& RpcMeta::method_name() const {
  // @@protoc_insertion_point(field_get:example.RpcMeta.method_name)
  return method_name_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void RpcMeta::set_method_name(const ::std::string& value) {
  
  method_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:example.RpcMeta.method_name)
}
inline void RpcMeta::set_method_name(const char* value) {
  
  method_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:example.RpcMeta.method_name)
}
inline void RpcMeta::set_method_name(const char* value, size_t size) {
  
  method_name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:example.RpcMeta.method_name)
}
inline ::std::string* RpcMeta::mutable_method_name() {
  
  // @@protoc_insertion_point(field_mutable:example.RpcMeta.method_name)
  return method_name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* RpcMeta::release_method_name() {
  // @@protoc_insertion_point(field_release:example.RpcMeta.method_name)
  
  return method_name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void RpcMeta::set_allocated_method_name(::std::string* method_name) {
  if (method_name != NULL) {
    
  } else {
    
  }
  method_name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), method_name);
  // @@protoc_insertion_point(field_set_allocated:example.RpcMeta.method_name)
}

// optional int32 data_size = 3;
inline void RpcMeta::clear_data_size() {
  data_size_ = 0;
}
inline ::google::protobuf::int32 RpcMeta::data_size() const {
  // @@protoc_insertion_point(field_get:example.RpcMeta.data_size)
  return data_size_;
}
inline void RpcMeta::set_data_size(::google::protobuf::int32 value) {
  
  data_size_ = value;
  // @@protoc_insertion_point(field_set:example.RpcMeta.data_size)
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace example

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_rpc_5fmeta_2eproto__INCLUDED
