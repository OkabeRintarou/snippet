import socket
import os
import struct
import threading
import time
from ctypes import *
from netaddr import IPNetwork, IPAddress


class IP(Structure):
	_fields_ = [
		('ihl', c_uint8, 4),
		('version', c_uint8, 4),
		('tos', c_uint8, 8),
		('len', c_uint16, 16),
		('id', c_uint16, 16),
		('offset', c_uint16, 16),
		('ttl', c_uint8, 8),
		('protocol_num', c_uint8, 8),
		('sum', c_uint16, 16),
		('src', c_uint32, 32),
		('dest', c_uint32, 32),
	]

	def __new__(self, socket_buffer = None):
		return self.from_buffer_copy(socket_buffer)

	def __init__(self, socket_buffer = None):
		self.protocol_map = { 1:'ICMP', 6:'TCP', 17:'UDP'}

		self.src_address = socket.inet_ntoa(struct.pack('<L', self.src))
		self.dest_address = socket.inet_ntoa(struct.pack('<L', self.dest))

		try:
			self.protocol = self.protocol_map[self.protocol_num]
		except:
			self.protocol = str(self.protocol_num)

class ICMP(Structure):
	_fields_ = [
		('type', c_uint8, 8),
		('code', c_uint8, 8),
		('checksum', c_uint16, 16),
		('unused', c_uint16, 16),
		('next_hop_mtu', c_uint16, 16),
	]

	def __new__(self, socket_buffer):
		return self.from_buffer_copy(socket_buffer)

	def __init__(self, sock_buffer):
		pass

host = '202.114.7.165'
subhost = '202.114.7.0/24'

# 自定义字符串, 在ICMP响应中进行核对
magic_message = 'PYTHONRULESi!'

def udp_sender(subnet, magic_message):
	time.sleep(5)
	sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

	for ip in IPNetwork(subhost):
		try:
			sender.sendto(magic_message.encode(), ('%s' % ip, 65212))
		except Exception as e:
			print(e)

if os.name == 'nt':
	socket_protocol = socket.IPPROTO_IP
else:
	socket_protocol = socket.IPPROTO_ICMP

sniffer = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket_protocol)

sniffer.bind((host, 0))

sniffer.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)

# 在Windows平台上启用混杂模式
if os.name == 'nt':
	sniffer.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)

t = threading.Thread(target = udp_sender, args=(subhost, magic_message))
t.start()

try:
	while True:
		raw_buffer = sniffer.recvfrom(65535)[0]
		ip_header = IP(raw_buffer[:20])
		#print('Protocol: %s %s -> %s' % (ip_header.protocol, ip_header.src_address, ip_header.dest_address))

		if ip_header.protocol == 'ICMP':
			offset = ip_header.ihl * 4
			buf = raw_buffer[offset:offset + sizeof(ICMP)]

			icmp_header = ICMP(buf)
			#print('ICMP: %s -> %s Type: %d Code:%d' % (ip_header.src_address, ip_header.dest_address, icmp_header.type, icmp_header.code))
			if icmp_header.code == 3 and icmp_header.type == 3:
				if IPAddress(ip_header.src_address) in IPNetwork(subhost):
					if raw_buffer[len(raw_buffer) - len(magic_message):] == magic_message.encode():
						print('Host Up: %s' % ip_header.src_address)

except Exception as e:
	print(e)
	# 关闭混杂模式
	if os.name == 'nt':
		sniffer.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)

