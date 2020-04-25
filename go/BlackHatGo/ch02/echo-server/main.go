package main

import (
	"bufio"
	"log"
	"net"
)

func echo(conn net.Conn) {
	defer conn.Close()

	reader := bufio.NewReader(conn)
	writer := bufio.NewWriter(conn)
	for {
		s, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalln("Unable to read data")
		}
		log.Printf("Received %d bytes: %s\n", len(s), s)

		// Send data via connection. Write
		log.Println("Writing data")
		if _, err = writer.WriteString(s); err != nil {
			log.Fatalln("Unable to write")
		}
		writer.Flush()
	}
}

func main() {
	listener, err := net.Listen("tcp", ":20080")
	if err != nil {
		log.Fatalln("Unable to bind to port 20080")
	}
	log.Println("Listening on 0.0.0.0:20080")
	for {
		conn, err := listener.Accept()
		log.Println("Received connection")
		if err != nil {
			log.Fatalln("Unable to accept connection")
		}
		go echo(conn)
	}
}
