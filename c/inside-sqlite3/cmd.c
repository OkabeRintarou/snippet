#include <sqlite3.h>
#include <stdio.h>

static int callback(void *NotUsed, int argc, char **argv, char **colName) {
  int i;
  for (i = 0; i < argc; i++) {
    printf("%s = %s\n", colName[i], argv[i] ? argv[i] : "NULL");
  }
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s DATABASE-FILE-NAME SQL_STATEMENT\n", argv[0]);
    return -1;
  }

  sqlite3 *db = NULL;
  char *zErrMsg = NULL;
  int rc;

  rc = sqlite3_open(argv[1], &db);
  if (rc != SQLITE_OK) {
    sqlite3_close(db);
    fprintf(stderr, "Could not open database: %s\n", sqlite3_errmsg(db));
    return rc;
  }

  rc = sqlite3_exec(db, argv[2], callback, 0, &zErrMsg);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "SQL error: %s\n", zErrMsg);
  }
  sqlite3_close(db);
  return 0;
}