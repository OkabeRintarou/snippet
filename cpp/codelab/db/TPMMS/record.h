#pragma once

#define ROW_SIZE 128
#define CONTENT_SIZE (ROW_SIZE - sizeof(int))

// each record takes up 100 bytes
struct record {
  int key;  // primary key
  char content[CONTENT_SIZE + 1];
};

record *random_generate_record();


