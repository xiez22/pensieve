import os

print("Please use this script AFTER build!!!")
print("You should first install apache2 and node.js!")
print("You can build with 'cd dash.js && npm install --force && bash compile.sh'!")

# copy the webpage files to /var/www/html
os.system("sudo cp video_server/*.html /var/www/html")
os.system("sudo cp video_server/dash.all.*.js /var/www/html")
os.system("sudo cp -r video_server/video* /var/www/html")
os.system("sudo cp video_server/Manifest.mpd /var/www/html")
os.system("sudo cp video_server/model.onnx /var/www/html")

# Run simple abr server
print('Starting ABR Server...')
os.system("python3 -m rl_server.simple_server")
