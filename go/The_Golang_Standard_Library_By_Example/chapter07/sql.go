package main

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

func main() {
	db, err := sql.Open("mysql", "root:123321@tcp(localhost:3306)/performance_schema?charset=utf8")
	if err != nil {
		panic(err)
	}
	defer db.Close()
	err = db.Ping()
	if err != nil {
		panic(err)
	}

	rows, err := db.Query("select user,host,current_connections,total_connections from accounts")
	if err != nil {
		panic(err)
	}

	defer rows.Close()

	for rows.Next() {
		var user, host *string
		var current_connections, total_connections int
		err = rows.Scan(&user, &host, &current_connections, &total_connections)
		if err != nil {
			panic(err)
		}
		if user != nil && host != nil {
			fmt.Println(*user, *host, current_connections, total_connections)
		} else if user == nil && host == nil {
			fmt.Println("<nil>", "<nil>", current_connections, total_connections)
		} else if host == nil {
			fmt.Println(*user, "<nil>", current_connections, total_connections)
		} else {
			fmt.Println("<nil>", *host, current_connections, total_connections)
		}
	}
	err = rows.Err()
	if err != nil {
		panic(err)
	}

	{
		// func (*DB) QueryRow()
		var user, host *string
		var current_connections, total_connections int
		err = db.QueryRow("select user,host,current_connections,total_connections from accounts where total_connections > ?", 2).Scan(&user, &host, &current_connections, &total_connections)
		if err != nil {
			panic(err)
		}

		if user != nil && host != nil {
			fmt.Println(*user, *host, current_connections, total_connections)
		} else if user == nil && host == nil {
			fmt.Println("<nil>", "<nil>", current_connections, total_connections)
		} else if host == nil {
			fmt.Println(*user, "<nil>", current_connections, total_connections)
		} else {
			fmt.Println("<nil>", *host, current_connections, total_connections)
		}
	}

	testUpdate()
	testMaxIdleConns()
}

func testUpdate() {
	for i := 0; i < 2; i++ {
		fmt.Println()
	}

	db, err := sql.Open("mysql", "root:123321@tcp(localhost:3306)/test?charset=utf8")
	if err != nil {
		panic(err)
	}
	defer db.Close()

	rows, err := db.Query("SELECT * FROM student")
	if err != nil {
		panic(err)
	}

	for rows.Next() {
		var name string
		var age int
		err = rows.Scan(&name, &age)
		if err != nil {
			panic(err)
		}
		fmt.Println("name:", name, ",age", age)
	}

	err = rows.Err()
	if err != nil {
		panic(err)
	}
	stmt, err := db.Prepare("INSERT INTO student VALUES(?,?)")
	if err != nil {
		panic(err)
	}

	res, err := stmt.Exec("lisi", 25)
	if err != nil {
		panic(err)
	}
	lastId, err := res.LastInsertId()
	if err != nil {
		panic(err)
	}
	rowCnt, err := res.RowsAffected()
	if err != nil {
		panic(err)
	}
	fmt.Printf("ID = %d, affected = %d\n", lastId, rowCnt)
}

func testMaxIdleConns() {
	db, err := sql.Open("mysql", "root:123321@tcp(localhost:3306)/test/?charset=utf8")
	if err != nil {
		panic(err)
	}
	db.SetMaxIdleConns(3)

	for i := 0; i < 10; i++ {
		go func() {
			db.Ping()
		}()
	}
	time.Sleep(20 * time.Second)
}
