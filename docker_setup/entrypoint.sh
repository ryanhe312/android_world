#!/bin/bash

# Start Emulator
#============================================
./docker_setup/start_emu_headless.sh && \
adb root && \
venv/bin/python -m server.android_server
