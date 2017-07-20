// Channel代表了不同goroutine之间的通信
// 通过Channel(通信)来使得不同goroutine来访问共享内存中的元素
package main

import (
	"log"
	"net/http"
	"time"
	)

const (
	numPollers		= 2
	pollInterval	= 60 * time.Second
	statusInterval	= 10 * time.Second
	errTimeout		= 10 * time.Second
)

var urls = []string {
	"http://www.zhihu.com",
	"http://www.biying.com",
	"http://www.baidu.com",
}

type State struct {
	url string
	status string
}

func StateMonitor(updateInterval time.Duration)chan<-State {
	updates := make(chan State)
	urlStatus := make(map[string]string)
	ticker := time.NewTicker(updateInterval)
	go func() {
		for {
			select {
			case <-ticker.C:
				logState(urlStatus)
			case s := <-updates:
				urlStatus[s.url] = s.status
			}
		}
	}()
	return updates;
}

func logState(s map[string]string) {
	log.Println("Current state:")
	if len(s) == 0 {
		log.Println("<empty>")
		return
	}
	for k,v := range s {
		log.Printf(" %s %s",k,v)
	}
}

type Resource struct {
	url			string
	errCount	int
}

func (r *Resource) Poll() string {
	resp,err := http.Head(r.url)
	if err != nil {
		log.Println("Error",r.url,err)
		r.errCount++
		return err.Error()
	}
	r.errCount = 0
	return resp.Status
}

func (r *Resource) Sleep(done chan<- *Resource) {
	time.Sleep(pollInterval + errTimeout * time.Duration(r.errCount))
	done <- r // 再次转移资源所有权
}

// 从chan中读取数据相当于资源从sender方转移到receiver方
func Poller(in <-chan *Resource,out chan<- *Resource,status chan<- State) {
	for r := range in {
		s := r.Poll() // 该goroutine拥有该资源的所有权
		status <- State{r.url,s}
		out <- r // 将资源所有权转交给receiver方
	}
}

func main() {
	pending,complete := make(chan *Resource),make(chan *Resource)
	status := StateMonitor(statusInterval)

	for i := 0;i < numPollers;i++ {
		go Poller(pending,complete,status)
	}

	go func() {
		for _,url := range urls {
			pending <- &Resource{url,0} // 转移资源所有权给Poller所在的goroutine
		}
	}()

	for r := range complete { // 重新取回资源所有权
		go r.Sleep(pending)
	}
}
