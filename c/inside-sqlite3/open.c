#include <sqlite3.h>
#include <stdio.h>

int main() {
  sqlite3 *db = NULL;
  sqlite3_stmt *stmt = NULL;
  int retcode;

  retcode = sqlite3_open("MyDB", &db); /* Open a database named MyDB */
  if (retcode != SQLITE_OK) {
    sqlite3_close(db);
    fprintf(stderr, "Could not open MyDB: %s\n", sqlite3_errstr(retcode));
    return retcode;
  }

  retcode = sqlite3_prepare(db, "select SID from students order by SID", -1,
                            &stmt, 0);
  if (retcode != SQLITE_OK) {
    sqlite3_close(db);
    fprintf(stderr, "Could not execute SELECT: %s\n", sqlite3_errstr(retcode));
    return retcode;
  }

  while (sqlite3_step(stmt) == SQLITE_ROW) {
    int i = sqlite3_column_int(stmt, 0);
    printf("SID = %d\n", i);
  }
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return SQLITE_OK;
}