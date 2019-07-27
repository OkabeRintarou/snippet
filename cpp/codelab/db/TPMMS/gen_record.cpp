#include <cstdio>
#include <memory>
#include "record.h"

#include <gflags/gflags.h>

DEFINE_uint32(record_numbers, 10000000, "generate record numbers");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 2) {
    fprintf(stderr, "usage: %s <filename>\n", argv[0]);
    return -1;
  }

  std::unique_ptr<FILE, void (*)(FILE *)> fp(std::fopen(argv[1], "wb+"),
                                             [](FILE *file) -> void {
                                               std::fflush(file);
                                               std::fclose(file);
                                             });
  if (!fp) {
    fprintf(stderr, "open file %s failed\n", argv[1]);
    return -1;
  }

  for (auto i = 0u; i < FLAGS_record_numbers; i++) {
    auto r = std::unique_ptr<record>(random_generate_record());
    fwrite(&r->key, sizeof(r->key), 1, fp.get());
    fwrite(r->content, sizeof(char), CONTENT_SIZE, fp.get());
  }
  return 0;
}