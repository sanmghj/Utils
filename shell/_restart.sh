#!/bin/bash
# sudo chmod +x _restart.sh
# Restart services defined in the SERVICES array

SERVICES=("nginx.service")
for SVC in "${SERVICES[@]}"; do
    echo "Stopping $SVC"
    sudo systemctl stop "$SVC"
done
for SVC in "${SERVICES[@]}"; do
    echo "Starting $SVC"
    sudo systemctl start "$SVC"
done