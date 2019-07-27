#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <queue>
#include <algorithm>
#include "record.h"
#include "timer.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

static const uint32_t _1K = 1024;
static const uint32_t _1M = _1K * _1K;

DEFINE_uint32(block_size, 4 * _1K, "block size");
DEFINE_uint32(total_memory, 10 * _1M, "total available memory bytes of DBMS");
DEFINE_string(sub_table_prefix, "sub_table_", "prefix of the file name of sub table");
DEFINE_bool(check_is_sorted, true, "check output keys are sorted");

static void save_record(FILE *fp, const record &r) {
  fwrite(&r.key, sizeof(r.key), 1, fp);
  fwrite(r.content, sizeof(char), CONTENT_SIZE, fp);
}

static bool read_record(FILE *fp, record *r) {
  assert(r);

  if (feof(fp)) {
    return false;
  }

  if (fread(&r->key, sizeof(r->key), 1, fp) != 1) {
    return false;
  }

  if (fread(r->content, sizeof(char), CONTENT_SIZE, fp) != CONTENT_SIZE) {
    return false;
  }
  r->content[CONTENT_SIZE] = '\0';
  return true;
}

class SubTableReader {
public:
  SubTableReader() {
    _file = nullptr;
    _all_read = true;
    _index = _record_nums = 0;
  }

  ~SubTableReader();

  bool init(const std::string &sub_table_filename);

  bool has_next() const;

  record next();

private:
  void read_records();

private:
  bool _all_read; // true if all records have read from this sub table
  FILE *_file;
  int _index;
  int _record_nums;
  std::vector<record> _records;
};

inline SubTableReader::~SubTableReader() {
  if (!_file) {
    std::fclose(_file);
  }
}

bool SubTableReader::init(const std::string &sub_table_filename) {
  _file = std::fopen(sub_table_filename.c_str(), "rb");
  if (!_file) {
    return false;
  }

  if (_file) {
    _records.resize(FLAGS_block_size / ROW_SIZE);
    read_records();
  }
  return true;
}

void SubTableReader::read_records() {
  _index = 0;
  _record_nums = 0;
  for (auto &r : _records) {
    if (!read_record(_file, &r)) {
      break;
    }
    ++_record_nums;
  }

  _all_read = _record_nums == 0;
}

bool SubTableReader::has_next() const {
  if (_all_read) {
    return false;
  }
  return _index < _record_nums;
}

record SubTableReader::next() {

  assert(_index < _record_nums);
  auto r = _records[_index];
  _index++;
  if (_index == _record_nums) {
    read_records();
  }
  return r;
}

class Sorter {
public:
  explicit Sorter(const std::vector<std::string> &sub_table_filenames);

  bool has_next() const;

  record next();

private:
  struct Record {
    Record(record a, SubTableReader *b) : r(a), reader(b) {}

    record r;
    SubTableReader *reader;
  };

  struct RecordComparer {
    bool operator()(const Record &lhs, const Record &rhs) const {
      return lhs.r.key > rhs.r.key;
    }
  };

private:
  std::unique_ptr<SubTableReader[]> _sub_tables;
  std::priority_queue<Record, std::vector<Record>, RecordComparer> _records_queue;
};

Sorter::Sorter(const std::vector<std::string> &sub_table_filenames) : _records_queue(Sorter::RecordComparer()) {

  _sub_tables = std::unique_ptr<SubTableReader[]>(new SubTableReader[sub_table_filenames.size()]);
  for (auto i = 0ul, e = sub_table_filenames.size(); i < e; i++) {
    if (!_sub_tables[i].init(sub_table_filenames[i])) {
      LOG(ERROR) << "open sub table " << sub_table_filenames[i] << " failed.";
    }
    if (_sub_tables[i].has_next()) {
      _records_queue.push(Record(_sub_tables[i].next(), &_sub_tables[i]));
    }
  }
}

bool Sorter::has_next() const {
  return !_records_queue.empty();
}

record Sorter::next() {
  auto inner_record = _records_queue.top();
  _records_queue.pop();
  if (inner_record.reader->has_next()) {
    _records_queue.push(Record(inner_record.reader->next(), inner_record.reader));
  }
  return inner_record.r;
}


std::vector<std::string> phase1(const char *const table_name) {
  // read all  tuples into memory, sort and save into sub-table.
  std::vector<record> records;
  std::unique_ptr<FILE, void (*)(FILE *)> fp(std::fopen(table_name, "rb"), [](FILE *fp) { fclose(fp); });
  if (!fp) {
    LOG(ERROR) << "open table file " << table_name << " failed.";
  }

  record r;
  auto current_memory = 0u;
  bool flag = true;
  int file_index = 0;

  Timer timer;
  timer.reset();

  std::vector<std::string> sub_table_filenames;
  do {
    while (current_memory + ROW_SIZE < FLAGS_total_memory && (flag = read_record(fp.get(), &r))) {
      records.emplace_back(r);
      current_memory += ROW_SIZE;
    }

    // save records to sub-table
    if (!records.empty()) {
      std::sort(records.begin(), records.end(),
                [](const record &lhs, const record &rhs) {
                  return lhs.key < rhs.key;
                });

      std::stringstream ss;
      ss << FLAGS_sub_table_prefix << file_index;
      const std::string &sub_table_filename(ss.str());

      std::unique_ptr<FILE, void (*)(FILE *fp)> out(std::fopen(sub_table_filename.c_str(), "wb+"),
                                                    [](FILE *fp) { fclose(fp); });
      if (!out) {
        LOG(ERROR) << "open file " << sub_table_filename << " failed.";
      }
      for (const auto &record : records) {
        save_record(out.get(), record);
      }
      sub_table_filenames.emplace_back(sub_table_filename);
    }

    file_index++;
    current_memory = 0;
    records.clear();
  } while (flag);

  printf("\tphase 1: %lf (ms)\n", timer.elapsed());
  return sub_table_filenames;
}

void phase2(const std::vector<std::string> &filenames) {
  Sorter sorter(filenames);
  if (FLAGS_check_is_sorted) {
    std::vector<int> keys;
    while (sorter.has_next()) {
      const auto &r = sorter.next();
      keys.push_back(r.key);
    }
    assert(std::is_sorted(keys.begin(), keys.end()));
  } else {
    Timer timer;
    timer.reset();
    while (sorter.has_next()) {
      sorter.next();
    }
    printf("\tphase 2: %lf (ms)\n", timer.elapsed());
  }
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 2) {
    fprintf(stderr, "usage: %s <table-name>\n", argv[0]);
    return -1;
  }
  assert((FLAGS_block_size % ROW_SIZE) == 0);

  printf("block size: %u (bytes), total memory: %u (bytes)\n", FLAGS_block_size, FLAGS_total_memory);
  // phase1
  const auto &sub_table_filenames = phase1(argv[1]);

  // phase2
  phase2(sub_table_filenames);

  return 0;
}