# Assignment 6 – Ray Tracer

**Name**: Yi-Cheng Hsiao (Nelson)
**UID**: 434006564
**The highest task you’ve completed:** 
Task1~Task3 & Task5~6 Done 
Task4 Partially Done 

Four Bonus Task:
*    **Multithreading** - process the image pixels in parallel.  
*    **Antialiasing** by generating multiple primary rays per pixel and averaging.
*    **Add Box** 
*    **Motion blur** - have a moving object, and average the result of multiple renders.
### Setting
*    Using **mac os** terminal to use command for compiling
*    Using **glm** library as math computation 
### vairable
* **Ray:** Include Ray origin & Direction
* **Light:** Light include the position, color and intensity 
* **Material:** Include each material parameter and reflective parameter 
* **Model:** It's a template for whole shape, which includes default **constructor** and a virtual function of **intersect**.
* **Camera:** It's view point setting, based on the homework requirement and generate ray to calculate the ray on image plane
* **Sphere:** The sphere model seeting & intersect function
* **Ellipsoid:** The Ellipsoid model seeting & intersect function
* **Box:** The box model seeting & intersect function
* **Move_Sphere:** The Move_Sphere model seeting & intersect function for motion blur
* **Plane:** The Plane model seeting & intersect function
* **Vertice:** Store the triangle's point position / normal
* **Triangle:** The sphere model seeting & intersect function
* **IntersectionResult:** Store the which part is the triangle hit point
* **BoundingSphere:** To detect the bouding of the obj model and speedup the hit computation for obj model

### Function

### Task 1: Camera Setup
* Implement it in camera class 
* Transfer the coordination from the world coordination to pixel coordination
* Deal with scene 8 task, implement the other function of coordination mapping 
### Task 2: Spheres and Shadows (Scenes 1 & 2)
* create the light and image plane setting 
* initialize the parameter of material and model 
* call the ray generate to generate each ray 
* create an image plane 
* Traverse each pixel on image plane 
* Do the ray_tracing function
### Task 3: Ellipsoid (Scene 3)
* The task is similiar to task2
* create the light and image plane setting 
* initialize the parameter of material and model 
* call the ray generate to generate each ray 
* create an image plane 
* Traverse each pixel on image plane 
* Do the ray_tracing function
### Task 4: Reflections (Scenes 4 & 5)
* The task is similiar to task2
* create the light and image plane setting 
* initialize the parameter of material and model 
* call the ray generate to generate each ray 
* create an image plane 
* Traverse each pixel on image plane 
* Do the ray_tracing function
* Special point is that using recursive to call the ray tracing function when the depth is not zero

### Task 6: Triangle Mesh (Scenes 6 & 7)
* rayIntersectsBoundingVolume, First test against the bounding volume
* Then test against each triangle in the mesh
* If an intersection was found, shade the pixel
* Reference intersection code from Internet 
* Calculate color uisng normal coordination 

### Task 7: Transforming the Camera (Scene 8)
* Change the setting of the camera
* Follow the task 1, and change the task7 parameter 
* Call the camera again, but spcifically to have a function for task8 coordination mapping 

### Bonus - Antialiasing with Motion blur (Scene 9)
* Use the same setting of Scene 2
* Change the pixel setting 
* 4x supersampling (2x2 grid per pixel)
* ```int samples_per_pixel = 4;  // 4x supersampling (2x2 grid per pixel)```
* ```float inv_samples = 1.0f / samples_per_pixel;```
* color /= (samples_per_pixel);  Average the color
* For the motion blur
* Add a sphere
* Move the sphere in a specific direction
* calsulate color in each time frame
* render it 
### Bonus - Multithreading with box: (Scene 0)
* Box class follow the Internet source 
* Using C++ multithreading 
* include the thread library: #include <thread>
* const int num_threads = std::thread::hardware_concurrency(); Get the thread number 
* create the thread vector 
* int rows_per_thread = height / num_threads;
```
for (int i = 0; i < num_threads; ++i) {
        int start_y = i * rows_per_thread;
        int end_y = (i + 1) == num_threads ? height : start_y + rows_per_thread;
        threads[i] = std::thread(render_section, start_y, end_y, width, std::ref(camera), &image, std::ref(models), std::ref(lights));
    }

    for (auto& t : threads) {
        t.join();
    }
```
                                
### Command control
* Scene 9:  Bonus for Antialiasing with Motion blur
* Scene 0:  Add Box with multi threading
                                