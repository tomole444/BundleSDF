python run_custom.py --mode run_video --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/2022-11-18-15-10-24_milk --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/out --use_segmenter 1 --use_gui 1 --debug_level 2
python run_custom.py --mode run_video --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/ownFritz2 --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/ownFritz --use_segmenter 0 --use_gui 1 --debug_level 4

lrwxrwxrwx 1 root root   26 Oct  4 04:28 g++ -> x86_64-conda-linux-gnu-g++
lrwxrwxrwx 1 root root   26 Oct  4 04:28 c++ -> x86_64-conda-linux-gnu-c++

#Änderungen permanent machen
sudo docker commit afc047950173 nvcr.io/nvidian/bundlesdf:latest
sudo docker ps -a
sudo docker images

#BundleTrack neu builden
cd BundleTrack/build && make -j11 && cd ../..


#Error "no GLIBCXX_3.4.29"
conda install scipy


#Error "no module grind_encoder"
cd mycuda/
python setup.py build_ext --inplace
export PYTHONPATH=$PYTHONPATH:/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/mycuda
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/py38/lib/python3.8/site-packages/torch/lib #?? keine Ahnung ob benötigt

#Error "no GLIBCXX_3.4.30"
conda install gcc=12.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/py38/x86_64-conda-linux-gnu/lib/



python run_custom.py --mode global_refine --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/ownFritz2 --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/ownFritz   # Change the path to your video_directory

#Visualisierung
python run_custom.py --mode draw_pose --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/ownFritz2 --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/ownFritz
