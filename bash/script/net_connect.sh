#!/bin/bash
iptables -F
iptables -P INPUT ACCEPT
iptables -P FORWARD ACCEPT
# enp0s25是宿主机的网络接口名
iptables -t nat -A POSTROUTING -o enp0s25 -j MASQUERADE

