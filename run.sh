export GLM_INCLUDE_DIR=/Users/yicheng/Desktop/A2/glm
#export GLFW_DIR=/Users/yicheng/Desktop/A2/glfw
#export GLEW_DIR=/Users/yicheng/Desktop/A2/glew
rm -r -f build
mkdir build
cd build
cmake ..
make
#./A4 ../resources
/Users/yicheng/Desktop/C_Raytracer/build/A6 1 512 output.jpg
