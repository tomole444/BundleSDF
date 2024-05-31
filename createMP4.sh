export FPS=15
ffmpeg -f image2 -r $FPS -i "/home/grass/Documents/Leyh/BundleSDF/ooutBookComb/pose_vis/%05d.png" -b:v 40M -b:a 192k -vcodec mpeg4 -y /home/grass/Documents/Leyh/BundleSDF/outBookCombViz.mp4

export FPS=15
ffmpeg -f image2 -r $FPS -i "/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoOrig/pose_vis/%05d.png" -b:v 40M -b:a 192k -vcodec mpeg4 -y /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoOrig.mp4