#include <dirent.h>
#include <limits.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>

/* function type that is called for each filename */
typedef int Myfunc(const char *, const struct stat *, int);

static Myfunc myfunc;
static int myftw(char *, Myfunc *);
static int dopath(Myfunc *);

static long nreg, ndir, nblk, nchr, nfifo, nslink, nsock, ntot;

int main(int argc, char *argv[]) {
  int ret;
  if (argc != 2) {
    fprintf(stderr, "usage: ftw <starting-pathname>");
    exit(1);
  }
  ret = myftw(argv[1], myfunc);
  ntot = nreg + ndir + nblk + nchr + nfifo + nslink + nsock;
  if (ntot == 0)
    ntot = 1;

  printf("regular files  = %7ld, %5.2f %%\n", nreg, nreg * 100.0 / ntot);
  printf("directories    = %7ld, %5.2f %%\n", ndir, ndir * 100.0 / ntot);
  printf("block special  = %7ld, %5.2f %%\n", nblk, nblk * 100.0 / ntot);
  printf("char special   = %7ld, %5.2f %%\n", nchr, nchr * 100.0 / ntot);
  printf("FIFOs          = %7ld, %5.2f %%\n", nfifo, nfifo * 100.0 / ntot);
  printf("symbolic links = %7ld, %5.2f %%\n", nslink, nslink * 100.0 / ntot);
  printf("socks          = %7ld, %5.2f %%\n", nsock, nsock * 100.0 / ntot);

  return 0;
}

#define FTW_F 1
#define FTW_D 2
#define FTW_DNR 3
#define FTW_NS 4

static char *fullpath;
static size_t pathlen;

static int myftw(char *pathname, Myfunc *func) {
  fullpath = (char *)malloc(sizeof(char) * (PATH_MAX + 1));
  pathlen = PATH_MAX;

  if (pathlen <= strlen(pathname)) {
    pathlen = strlen(pathname) * 2;
    if ((fullpath = realloc(fullpath, pathlen)) == NULL) {
      perror("realloc failed");
      free(fullpath);
      exit(1);
    }
  }
  strcpy(fullpath, pathname);
  return dopath(func);
}

static int dopath(Myfunc *func) {
  struct stat statbuf;
  struct dirent *dirp;
  DIR *dp;
  int ret, n;

  if (lstat(fullpath, &statbuf) < 0) {
    return func(fullpath, &statbuf, FTW_NS);
  }
  if (S_ISDIR(statbuf.st_mode) == 0) { /* not a directory */
    return func(fullpath, &statbuf, FTW_F);
  }

  /* It's a directory. First call func() for the directory,
   * then process each filename in the directory.
   */
  if ((ret = func(fullpath, &statbuf, FTW_D)) != 0) {
    return ret;
  }
  n = strlen(fullpath);
  if (n + NAME_MAX + 2 > pathlen) {
    pathlen *= 2;
    if ((fullpath = realloc(fullpath, pathlen)) == NULL) {
      perror("realloc error");
      exit(1);
    }
  }
  fullpath[n++] = '/';
  fullpath[n] = 0;

  if ((dp = opendir(fullpath)) == NULL) {
    return func(fullpath, &statbuf, FTW_DNR);
  }

  while ((dirp = readdir(dp)) != NULL) {
    if (strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0) {
      continue;
    }
    strcpy(&fullpath[n], dirp->d_name);
    if ((ret = dopath(func)) != 0) {
      break;
    }
  }

  fullpath[n - 1] = 0;
  if (closedir(dp) < 0) {
    fprintf(stderr, "can't close diretory %s", fullpath);
    return 1;
  }
  return ret;
}

static int myfunc(const char *pathname, const struct stat *statptr, int type) {
  switch (type) {
  case FTW_F:
    switch (statptr->st_mode & S_IFMT) {
    case S_IFREG:
      nreg++;
      break;
    case S_IFBLK:
      ndir++;
      break;
    case S_IFCHR:
      nblk++;
      break;
    case S_IFIFO:
      nfifo++;
      break;
    case S_IFLNK:
      nslink++;
      break;
    case S_IFSOCK:
      nsock++;
      break;
    case S_IFDIR:
      fprintf(stderr, "for S_IFDIR for %s", pathname);
    }
    break;
  case FTW_D:
    ndir++;
    break;
  case FTW_DNR:
    fprintf(stderr, "can't read directory %s", pathname);
    return 1;
  case FTW_NS:
    fprintf(stderr, "stat error for %s", pathname);
    return 1;
  default:
    fprintf(stderr, "unknow type %d for pathname %s", type, pathname);
    return 1;
  }
  return 0;
}
