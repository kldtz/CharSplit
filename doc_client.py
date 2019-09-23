#!/usr/bin/python3
# Copyright eBrevia.com 2019
# pylint: disable=missing-docstring, invalid-name
# A trivial client for doc_server
# Reads all of stdin, sends it to doc_server,
# splits, sends result to stdout
# This is equivalent to doc_split but with less overhead

import socket
import sys

import doc_config

if len(sys.argv) > 2:
    host = sys.argv[2]
else:
    host = doc_config.DEFAULT_HOST
if len(sys.argv) > 1:
    port = int(sys.argv[1])
else:
    port = doc_config.DEFAULT_PORT

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect((host, port))
    print("connected to server", file=sys.stderr)
    input_str = sys.stdin.read()
    print("input read", file=sys.stderr)
    input_bytes = input_str.encode()
    client.sendall(input_bytes)
    print("input sent", file=sys.stderr)
    client.shutdown(socket.SHUT_WR)
    print("shut down write side", file=sys.stderr)

    output_bytes = bytearray(b'')
    blockno = 0
    while True:
        data = client.recv(2048)
        if not data:
            break
        #print("read block %d" % (blockno, file=sys.stderr))
        blockno += 1
        output_bytes += data
    print("finished reading", file=sys.stderr)
    output_str = output_bytes.decode()
    sys.stdout.write(output_str)
