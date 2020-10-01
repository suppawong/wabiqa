#!/usr/bin/env bash
service nginx start
gunicorn -w 2 --timeout 9000 app:app 