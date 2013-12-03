all: $(SOURCES)
	#g++ -std=gnu++0x -I./include src/transitionModel.cpp -L./lib -ltrng4 -o tm
	g++ -O2 -msse4.2 -fopenmp -std=gnu++0x `pkg-config --cflags opencv` -I./include src/ImageHelper.cpp src/parFilter.cpp src/tracker.cpp -L./lib -ltrng4 `pkg-config --libs opencv` -o pf

clean:
	rm

