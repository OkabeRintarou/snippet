#include <cmath>
#include "record.h"

static void gen_random(char *s, const int len) {
  static const char alphanum[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";

  for (int i = 0; i < len; ++i) {
    s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
  }
  s[len] = 0;
}

record *random_generate_record() {
  auto *r = new record;
  r->key = rand();
  gen_random(r->content, CONTENT_SIZE);
  return r;
}