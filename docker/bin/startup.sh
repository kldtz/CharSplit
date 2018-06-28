#!/bin/bash
set -e

mkdir -p /var/log/ebrevia
chmod a+w /var/log/ebrevia

# read externally provided ca certs if available
if [ -f "/eb_files/config/ssl/ca.crt" ]; then
  echo "Activating custom CA, CA certificates found"
  cp /eb_files/config/ssl/ca.crt /usr/local/share/ca-certificates
  update-ca-certificates
fi

echo -e \
"╔═══════════════╗
║ e B R E V I A ║
╚═¯═════════════╝
Intelligent Contract Analytics

Copyright 2017 eBrevia, Inc."

