#!/bin/bash
# sudo kill -9 $(sudo lsof -t -i :8767)

python3 -m src.webui.server --host 127.0.0.1 --port 8767   