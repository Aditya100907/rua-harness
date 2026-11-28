#!/bin/bash
# Standing orders: at 7 hours into an SSH connection to Bouchet, alert Adi (macOS notification + spoken
# message) that Duo is needed. Checks every 10 minutes; alerts once per master connection.
#   nohup bash scripts/ssh_duo_alert.sh &
ALERTED=""
while true; do
  PID=$(ssh -O check bouchet 2>&1 | grep -o "pid=[0-9]*" | cut -d= -f2)
  if [ -n "$PID" ]; then
    ET=$(ps -o etime= -p $PID | tr -d ' ')          # [[dd-]hh:]mm:ss
    SECS=$(echo "$ET" | awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }')
    if [ "$SECS" -ge 25200 ] && [ "$ALERTED" != "$PID" ]; then
      osascript -e 'display notification "The Bouchet SSH connection is 7 hours old and expires in about 1 hour. Please run: ssh bouchet true, and approve Duo." with title "Duo needed for Bouchet" sound name "Glass"'
      say "Adi, Duo is needed for Bouchet. The connection expires in about one hour."
      echo "$(date) alerted (master pid $PID, age $ET)"
      ALERTED=$PID
    fi
  fi
  sleep 600
done
