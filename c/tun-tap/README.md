# Run

## shell 1

```bash
make
sudo ./tun
```

启动tun程序，程序会创建一个新的tun设备，该设备会阻塞在read系统调用等待数据包过来。

## shell 2

```bash
sudo tcpdump -i tun0
```

启动抓包程序，抓经过tun0的包

## shell 3

tun程序启动之后，通过ip link 命令就会发现系统多了个tun设备。

新的设备没有ip，先给tun设备添加IP地址：

```bash
sudo ip addr add 192.168.3.11/24 dev tun0
```

然后通过下面的命令将tun0启动起来：

```bash
sudo ip link set tun0 up
```

然后尝试ping 192.168.3.0/24网段的IP，根据默认路由，该包会走tun0设备，由于我们的程序中收到数据包后，啥都没干，相当于把数据包丢弃了，所以这里的ping根本收不到返回包。

