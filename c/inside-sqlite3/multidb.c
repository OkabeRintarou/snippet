#include <sqlite3.h>
#include <stdio.h>

int main() {
  sqlite3 *db = NULL;
  sqlite3_open("MyDB", &db);
  sqlite3_exec(db, "attach database MyDBExtn as DB1", 0, 0, 0);
  sqlite3_exec(db, "begin", 0, 0, 0);
  sqlite3_exec(db, "insert into students values(3000)", 0, 0, 0);
  sqlite3_exec(db, "insert into courses values('SQLite database',2000)", 0, 0,
               0);
  sqlite3_exec(db, "commit", 0, 0, 0);
  sqlite3_close(db);
  return 0;
}