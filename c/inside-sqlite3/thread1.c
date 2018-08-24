#include <pthread.h>
#include <sqlite3.h>
#include <stdio.h>
#include <unistd.h>

void *myInsert(void *arg) {
  sqlite3 *db = NULL;
  sqlite3_stmt *stmt = NULL;
  int val = (int)arg;
  char SQL[100];
  int rc;

  // comment next sleep call may cause some or all
  // INSERT statements fail due to lock conflict
  sleep((unsigned)val);

  rc = sqlite3_open("MyDB", &db);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "Thread[%d] fails to open the database\n", val);
    goto errReturn;
  }

  sprintf(SQL, "insert into students values(%d)", val);

  rc = sqlite3_prepare(db, SQL, -1, &stmt, 0);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "Thread[%d] fails to prepare SQL: "
                    "%s -> return code %d(%s)\n",
            val, SQL, rc, sqlite3_errstr(rc));
    goto errReturn;
  }

  rc = sqlite3_step(stmt);
  if (rc != SQLITE_DONE) {
    fprintf(stderr, "Thread[%d] fails to execute SQL: "
                    "%s -> return code: %d(%s)\n",
            val, SQL, rc, sqlite3_errstr(rc));
  } else {
    printf("Thread[%d] successfully executes SQL: %s\n", val, SQL);
  }

errReturn:
  sqlite3_close(db);
  return (void *)rc;
}

int main() {
  pthread_t threads[10];
  int i;
  for (i = 0; i < 10; i++) {
    pthread_create(&threads[i], NULL, myInsert, (void *)i);
  }

  for (i = 0; i < 10; i++) {
    pthread_join(threads[i], NULL);
  }
  return 0;
}