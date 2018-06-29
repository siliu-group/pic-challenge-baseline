export PATH := /usr/local/cuda-8.0/bin:$(PATH)

all: box_intersections lstm

box_intersections:
	cd lib/box_intersections_cpu; python3 setup.py build_ext --inplace
lstm:
	cd lib/lstm/highway_lstm_cuda; ./make.sh
