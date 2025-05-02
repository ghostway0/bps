all:
	cmake -Bbuild -GNinja -DCMAKE_CXX_COMPILER=g++
	cmake --build build

release:
	cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
	cmake --build build
