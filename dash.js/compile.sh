# filename=$1
rm -r dist/
grunt --force
cp ./dist/dash.all.min.js ../video_server/
cp ./dist/dash.all.debug.js ../video_server/
