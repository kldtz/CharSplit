#!/usr/bin/python3
# Copyright eBrevia.com 2019
"""Document splitting server."""

import socket
import sys

import doc_config
import doc_split

def run(client_socket, client_address, de_dict):
    """Reads, splits, writes."""
    input_bytes = bytearray(b'')
    blockno = 0
    while True:
        block = client_socket.recv(2048)
        #print ("Read block %d" % blockno, file=sys.stderr)
        blockno += 1
        if not block:
            break
        input_bytes += block
    print("Read %d bytes" % (len(input_bytes)), file=sys.stderr)
    input_str = input_bytes.decode()
    output_str = doc_split.doc_split(input_str, de_dict)
    output_bytes = output_str.encode()
    print("Split the text", file=sys.stderr)
    client_socket.sendall(output_bytes)
    print("Written %d bytes" % (len(output_bytes)), file=sys.stderr)
    client_socket.shutdown(socket.SHUT_WR)
    print("Client at ", client_address, " disconnecting", file=sys.stderr)

def main():
    """Main server program.
       Reads the whole of standard input, splits it all,
       sends the result to standard output.
       Usage:  doc_server dict port
    """
    if len(sys.argv) > 2:
        port = sys.argv[2]
    else:
        port = doc_config.DEFAULT_PORT
    if len(sys.argv) > 1:
        de_dict = sys.argv[1]
    else:
        de_dict = doc_config.DEFAULT_DICT

    doc_split.load_known_words(de_dict)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', port))
    print("Server started", file=sys.stderr)
    print("Waiting for client request..", file=sys.stderr)
    while True:
        server.listen(1)
        client, client_address = server.accept()
        run(client, client_address, de_dict)

if __name__ == "__main__":
    main()
