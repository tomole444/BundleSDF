python run_custom.py --mode run_video --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/2022-11-18-15-10-24_milk --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/out --use_segmenter 1 --use_gui 1 --debug_level 2
python run_custom.py --mode run_video --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/ownFritz2 --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/ownFritz --use_segmenter 0 --use_gui 1 --debug_level 4

python run_custom.py --mode run_video --video_dir /home/grass/Documents/Leyh/datasets/ownBuch --out_folder /home/grass/Documents/Leyh/BundleSDF/outBuch --use_segmenter 0 --use_gui 1 --debug_level 2

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

#Kompiliert nicht
(py38) root@fe-w404-u:/opt/conda/envs/py38/bin# ls | grep gcc
lrwxrwxrwx 1 root root   33 Nov  2 04:11 gcc-ranlib -> x86_64-conda-linux-gnu-gcc-ranlib
lrwxrwxrwx 1 root root   29 Nov  2 04:11 gcc-nm -> x86_64-conda-linux-gnu-gcc-nm
lrwxrwxrwx 1 root root   29 Nov  2 04:11 gcc-ar -> x86_64-conda-linux-gnu-gcc-ar
lrwxrwxrwx 1 root root   26 Nov  2 04:11 gcc-12 -> x86_64-conda-linux-gnu-gcc
lrwxrwxrwx 1 root root   26 Nov  2 04:11 cc -> x86_64-conda-linux-gnu-gcc
lrwxrwxrwx 1 root root   14 Nov 16 01:07 gcc -> /usr/bin/gcc-9                  #neu erstellt

#gcc: fatal error: cannot execute 'cc1plus': execvp: No such file or directory
find /usr -name "cc1plus"
ln -s /usr/lib/gcc/x86_64-linux-gnu/9/cc1plus /opt/conda/envs/py38/bin

#ModuleNotFoundError: No module named 'pynvml'
pip install pynvml

#complete rebuild
cd BundleTrack/ && rm -rf build && mkdir build && cd build && cmake .. && make -j11 && cd ../..
#rebuild
cd BundleTrack/build/ && cmake .. && make -j11 && cd ../..
#unsuccessful build 
cmake .. && make -j11 && cd ../..

# Fehler
ImportError: /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/BundleTrack/build/libBundleTrack.so: undefined symbol: _ZN11FeatureTreeC1Ev
-> kein Konstruktor definiert

#Log console to file
echo test 2>&1 | tee SomeFile.txt

python run_custom.py --mode global_refine --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/ownFritz2 --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/ownFritz   # Change the path to your video_directory

#Visualisierung
python run_custom.py --mode draw_bb --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/ownFritz2 --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/ownFritz

# Run BundleSDF to get the pose and reconstruction results
python run_ho3d.py --video_dirs /home/thws_robotik/Documents/Leyh/6dpose/datasets/HO3D_v3/evaluation/SM1 --out_dir /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outHO3D --use_gui 1 

# Benchmark the output results
python benchmark_ho3d.py --video_dirs /home/thws_robotik/Documents/Leyh/6dpose/datasets/HO3D_v3/evaluation/SM1 --out_dir /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outHO3D --log_dir /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outHO3Dlog

#Visualisierung Ho3D Pose
python run_custom.py --mode draw_pose --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/HO3D_v3/evaluation/SM1  --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outHO3D

#Visualisierung FritzCon
python run_custom.py --mode draw_pose --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/ownFritzConcat  --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outFritzCon

#Buch
python run_custom.py --mode run_video --video_dir /home/grass/Documents/Leyh/datasets/ownBook  --out_folder /home/grass/Documents/Leyh/BundleSDF/outBook --use_segmenter 0 --use_gui 1 --debug_level 1

python run_custom.py --mode global_refine --video_dir /home/grass/Documents/Leyh/datasets/ownBookSmall --out_folder /home/grass/Documents/Leyh/BundleSDF/outBookSmall --debug_level 4

python run_custom.py --mode run_video --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBuch --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/ownBuch --use_segmenter 0 --use_gui 1 --debug_level 1


#Truck
python run_custom.py --mode run_video --video_dir /home/grass/Documents/Leyh/datasets/ownTruckSmall --out_folder /home/grass/Documents/Leyh/BundleSDF/outTruckSmall --use_segmenter 0 --use_gui 1 --debug_level 4 2>&1 | tee TruckSmall.log

#Duplo
python run_custom.py --mode run_video --video_dir /home/grass/Documents/Leyh/datasets/ownDuploAz --out_folder /home/grass/Documents/Leyh/BundleSDF/outDuploAz --use_segmenter 0 --use_gui 1 --debug_level 4 2>&1 | tee log/DuploAz.log 
python run_custom.py --mode run_video --video_dir /home/grass/Documents/Leyh/datasets/ownDuploZed --out_folder /home/grass/Documents/Leyh/BundleSDF/outDuploZed --use_segmenter 0 --use_gui 1 --debug_level 4 2>&1 | tee log/DuploZed.log 

#Buch
python run_custom.py --mode run_video --video_dir /home/grass/Documents/Leyh/datasets/ownBookComb  --out_folder /home/grass/Documents/Leyh/BundleSDF/outBookComb --use_segmenter 0 --use_gui 1 --debug_level 4 2>&1 | tee log/BookComb.log

python run_realtime.py --mode run_video --video_dir /home/grass/Documents/Leyh/datasets/bookRealtime720p --key_folder /home/grass/Documents/Leyh/BundleSDF/outBookComb720p --out_folder /home/grass/Documents/Leyh/BundleSDF/outRealtime --use_segmenter 0 --use_gui 0 --debug_level 4

python run_custom.py --mode run_video --video_dir /home/grass/Documents/Leyh/datasets/ownBookComb  --out_folder /home/grass/Documents/Leyh/BundleSDF/outBookCombDEBUG --use_segmenter 0 --use_gui 0 --debug_level 4

#PVNet connector
python run_custom.py --mode run_video --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBuch --out_folder /home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/ownBuchTest --use_gui 0

python run_custom.py --mode run_video --video_dir /home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo --out_folder outBuchTest --use_gui 0