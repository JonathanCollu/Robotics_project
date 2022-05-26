#!/bin/bash

# how to run: ./upload_file.sh <IP address of the picar> <local filepath>

scp -o ProxyJump=pi@$1 $2 pi@picar01:
