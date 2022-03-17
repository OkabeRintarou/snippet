#include <cstdio>
#include "file_reader.h"

std::string file_read(const char *filename) {
  std::string content;
  FILE *file = fopen(filename, "rt");

  if (file) {
    fseek(file, 0, SEEK_END);
    auto count = ftell(file);
    rewind(file);
    if (count > 0) {
      content.resize(count + 1);
      count = fread(content.data(), sizeof(char), count, file);
      content[count] = '\0';
    }
    fclose(file);
  }
  return content;
}
