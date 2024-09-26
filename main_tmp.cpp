#define GLM_ENABLE_EXPERIMENTAL
#include <cassert>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <string>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "Image.h"


// This allows you to skip the `std::` in front of C++ standard library
// functions. You can also say `using std::cout` to be more selective.
// You should never do this in a header file.
#include <fstream>
#include <limits>
#include <cmath>
#include <algorithm>



using namespace std;
const float PI = 3.14159265358979323846;
const float INF = std::numeric_limits<float>::infinity();


class Ray{
	public:
		glm::vec3 origin;
		glm::vec3 direction;
	
		Ray(const glm::vec3 &origin, const glm::vec3 &direction) : origin(origin), direction(glm::normalize(direction)) {}
};

class Light{
	public:
		glm::vec3 Position;
		glm::vec3 color;
		float Intensity;
	
		Light(const glm::vec3 &position, float intensity) : Position(position), Intensity(intensity){
			color = glm::vec3(1.0, 1.0, 1.0);
		}
};

class Camera{
	public:
		glm::vec3 Position;
		glm::vec3 Rotation;
		float View; 
		float Aspect_ratio;
		int Image_width;
		int Image_height;
		float Fov;
	    glm::vec3 Foward;
		glm::vec3 Up;
		glm::vec3 Right;
		float ImagePlaneDistance;

		Camera(glm::vec3 position, glm::vec3 rotation, float view, float aspect_ratio, int image_width, int image_height)
		: Position(position), Rotation(rotation), View(glm::radians(view)), Aspect_ratio(aspect_ratio), 
		Image_width(image_width), Image_height(image_height),
		Foward(glm::vec3(0.0f, 0.0f, -1.0f)),  // CAMERA LOOKS TOWARDS -Z
		Up(glm::vec3(0.0f, 1.0f, 0.0f)),        // Y UP
		Right(glm::vec3(1.0f, 0.0f, 0.0f)),     // X RIGHT
		ImagePlaneDistance(1.0f){
			float image_P_Height = 2.0f * ImagePlaneDistance * tan(view / 2.0f);
			float image_P_Width  = image_P_Height * Aspect_ratio;

			Right *= image_P_Width; 
			Up    *= image_P_Height;
		}

	Ray generate_ray(int x, int y) const {
		
		// Image plane dimensions based on the FOV
		float halfHeight = tan(View / 2);
		float halfWidth = Aspect_ratio * halfHeight;

		// Pixel dimensions
		float pixelWidth = (halfWidth * 2) / Image_width;
		float pixelHeight = (halfHeight * 2) / Image_height;

		float u = halfWidth  - (x + 0.5f) * pixelWidth;
		float v = halfHeight - (y + 0.5f) * pixelHeight;

		// Ray direction from camera to pixel, in camera space
		//std::cout << u << ' ' << v << endl;
		glm::vec3 rayDir = glm::normalize(glm::vec3(u-Position.x, v-Position.y, -ImagePlaneDistance));
		glm::vec3 rayPos = Position;
		// return a ray
		return Ray(rayPos, rayDir);
    }
};

class Material{
	public:
		glm::vec3 Diffuse;
		glm::vec3 Specular;
		glm::vec3 Ambient;
		float Exponent;
		Material(const glm::vec3& diffuse, const glm::vec3& specular, const glm::vec3& ambient, float exponent): Diffuse(diffuse), Specular(specular), Ambient(ambient), Exponent(exponent){}
};

class Sphere{
	public:
		glm::vec3 Position;
		glm::vec3 Scale;
		glm::vec3 Rotation;
		Material materials;
		float radius;
		Sphere(const glm::vec3& position, const glm::vec3& scale, const glm::vec3& rotation, const Material& material): 
		Position(position), Scale(scale), Rotation(rotation), materials(material){
			radius = scale.x;
		};
};

/*

// Utility function to solve the quadratic equation used in the ray-sphere intersection test
bool solveQuadratic(const float& a, const float& b, const float& c, float& x0, float& x1) {
    float discr = b * b - 4 * a * c;
    if (discr < 0) return false;
    else if (discr == 0) x0 = x1 = -0.5 * b / a;
    else {
        float q = (b > 0) ?
            -0.5 * (b + std::sqrt(discr)) :
            -0.5 * (b - std::sqrt(discr));
        x0 = q / a;
        x1 = c / q;
    }
    if (x0 > x1) std::swap(x0, x1);
    return true;
}

class Ellipsoid {
public:
    glm::vec3 center, scale;
    Material material;

    Ellipsoid(const glm::vec3& center, const glm::vec3& scale, const Material& material)
        : center(center), scale(scale), material(material) {}

    // Ellipsoid intersection logic will be similar to sphere intersection, 
    // but you must apply the inverse scaling to the ray first
	bool intersect(const Ray& ray, float& t_near) {
			// Transform the ray to the ellipsoid's object space
			glm::mat4 transform = glm::scale(glm::mat4(1.0f), 1.0f / this->scale);
			glm::vec3 transformed_origin = glm::vec3(transform * glm::vec4(ray.origin - this->center, 1.0f));
			glm::vec3 transformed_direction = glm::vec3(transform * glm::vec4(ray.direction, 0.0f));

			Ray transformed_ray(transformed_origin, glm::normalize(transformed_direction));

			// Now we do a standard sphere intersection test in object space
			float t0, t1; // solutions for t if the ray intersects
			glm::vec3 L = -transformed_origin;
			float a = glm::dot(transformed_direction, transformed_direction);
			float b = 2.0 * glm::dot(transformed_direction, L);
			float c = glm::dot(L, L) - 1.0; // radius is 1.0 in the scaled space
			if (!solveQuadratic(a, b, c, t0, t1)) return false;

			if (t0 > t1) std::swap(t0, t1);

			if (t0 < 0) {
				t0 = t1; // if t0 is negative, let's use t1 instead
				if (t0 < 0) return false; // both t0 and t1 are negative
			}

			t_near = t0;

			// Transform the intersection point back to the world space
			glm::vec3 intersection_point = transformed_origin + transformed_direction * t_near;
			glm::mat4 inverse_transform = glm::scale(glm::mat4(1.0f), this->scale);
			intersection_point = glm::vec3(inverse_transform * glm::vec4(intersection_point, 1.0f)) + this->center;

			// Now we have the intersection point in world space
			// You may need the normal at the intersection point for shading calculations
			glm::vec3 normal = glm::normalize(glm::vec3(glm::transpose(glm::inverse(transform)) * glm::vec4(intersection_point - this->center, 0.0f)));

			return true;
	}
};

*/
class Plane {
public:
    glm::vec3 normal;
    glm::vec3 position;

    Plane(const glm::vec3& normal, const glm::vec3& position) : normal(glm::normalize(normal)), position(position) {}

    // Implement the ray-plane intersection method
    bool intersect(const Ray& ray, float& t) {
        // Calculate the denominator of the intersection formula
        float denominator = glm::dot(normal, ray.direction);
        
        // If the denominator is very close to 0, the ray is parallel to the plane
        if (std::abs(denominator) > 1e-6) {
            glm::vec3 p0_to_origin = position - ray.origin;
            t = glm::dot(p0_to_origin, normal) / denominator;
            
            // If t is negative, the intersection is behind the ray's origin
            return (t >= 0);
        }

        return false;
    }
};

// function
void task1_2(int width, int height, string filename);
//void task3(int width, int height, string filename);
//void task4(int width, int height, string filename);
//void task5(int width, int height, string filename);

int main(int argc, char **argv)
{
	if(argc < 2) {
		cout << "Usage: A1 meshfile" << endl;
		return 0;
	}
	//string meshName(argv[1]);

	int scene = atoi(argv[1]);
    // Width of image
    int image_size = atoi(argv[2]);
    // Height of image
    std::string file_name(argv[3]);

	// Load geometry
	vector<float> posBuf; // list of vertex positions
	vector<float> norBuf; // list of vertex normals
	vector<float> texBuf; // list of vertex texture coords

	/*
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	string errStr;
	bool rc = tinyobj::LoadObj(&attrib, &shapes, &materials, &errStr, meshName.c_str());
	if(!rc) {
		cerr << errStr << endl;
	} else {
		// Some OBJ files have different indices for vertex positions, normals,
		// and texture coordinates. For example, a cube corner vertex may have
		// three different normals. Here, we are going to duplicate all such
		// vertices.
		// Loop over shapes
		for(size_t s = 0; s < shapes.size(); s++) {
			// Loop over faces (polygons)
			size_t index_offset = 0;
			for(size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
				size_t fv = shapes[s].mesh.num_face_vertices[f];
				// Loop over vertices in the face.
				for(size_t v = 0; v < fv; v++) {
					// access to vertex
					tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
					posBuf.push_back(attrib.vertices[3*idx.vertex_index+0]);
					posBuf.push_back(attrib.vertices[3*idx.vertex_index+1]);
					posBuf.push_back(attrib.vertices[3*idx.vertex_index+2]);
					if(!attrib.normals.empty()) {
						norBuf.push_back(attrib.normals[3*idx.normal_index+0]);
						norBuf.push_back(attrib.normals[3*idx.normal_index+1]);
						norBuf.push_back(attrib.normals[3*idx.normal_index+2]);
					}
					if(!attrib.texcoords.empty()) {
						texBuf.push_back(attrib.texcoords[2*idx.texcoord_index+0]);
						texBuf.push_back(attrib.texcoords[2*idx.texcoord_index+1]);
					}
				}
				index_offset += fv;
				// per-face material (IGNORE)
				shapes[s].mesh.material_ids[f];
			}
		}
	}
	cout << "Number of vertices: " << posBuf.size()/3 << endl;
	*/
	switch(scene) {
        case 1:
            task1_2(image_size, image_size, file_name);
            break;
        case 2:
            task1_2(image_size, image_size, file_name);
			break;
		case 3:
			//task3(image_size, image_size, file_name);
		case 4:
			//task4(image_size, image_size, file_name);
		case 5:
			//task5(image_size, image_size, file_name);
        default:
            break;
    }

	return 0;
}


bool intersect(const Ray& ray, const Sphere& sphere, float& t) {
	
	// ray origin to the sphere's center pos 
    glm::vec3 oc = ray.origin - sphere.Position;

	// b*b - 4ac
    float a = glm::dot(ray.direction, ray.direction);
    float b = 2.0f * glm::dot(oc, ray.direction);
    float c = glm::dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;

	// < 0, no intersection
    if (discriminant < 0) {
        return false;
    } else {
	// >= 0, intersection one or multiple points 
		//  / 2a
        discriminant = std::sqrt(discriminant);
        float t0 = (-b - discriminant) / (2 * a);
        float t1 = (-b + discriminant) / (2 * a);
        
        if (t0 > t1) std::swap(t0, t1);
        if (t0 < 0) {
            t0 = t1; // If t0 is negative, let's use t1 instead
            if (t0 < 0) return false; // Both t0 and t1 are negative
        }

        t = t0;
		// if hit the sphere 
        return true;
    }
}

// Blinn-Phong shading model
glm::vec3 Blinn_Phong(const glm::vec3& intersection_point, const glm::vec3& normal, const Material& material, const Light& light, const glm::vec3& view_position) {
    glm::vec3 lightDir = glm::normalize(light.Position - intersection_point);
    glm::vec3 viewDir = glm::normalize(view_position - intersection_point);

    // Ambient reflection
    glm::vec3 ambient = material.Ambient * light.color * light.Intensity;

    // Diffuse reflection
    float diffuseFactor = std::max(glm::dot(normal, lightDir), 0.0f);
    glm::vec3 diffuse = material.Diffuse * light.color * diffuseFactor * light.Intensity;

    // Specular reflection
    glm::vec3 halfwayDir = glm::normalize(lightDir + viewDir);
    float specularFactor = std::pow(std::max(glm::dot(normal, halfwayDir), 0.0f), material.Exponent);
    glm::vec3 specular = material.Specular * light.color * specularFactor * light.Intensity;

    return ambient + diffuse + specular;
}



bool is_in_shadow(const glm::vec3& point, const Light& light, const std::vector<Sphere>& spheres, const Sphere& ignore_sphere) {
	glm::vec3 to_light = glm::normalize(light.Position - point);
    Ray shadow_ray(point, to_light);
    float t_light = glm::length(light.Position - point);

    for (const Sphere& sphere : spheres) {
        if (&sphere == &ignore_sphere) continue;
        float t = 0;
        if (intersect(shadow_ray, sphere, t) && t < t_light) {
            return true; // The point is in shadow as another sphere is closer to the light
        }
    }
    return false;
}

glm::vec3 trace_ray(const Ray& ray, const std::vector<Sphere>& spheres, const Light& light) {
    // back ground color 
	glm::vec3 color(0.0, 0.0, 0.0); 
    float t_min = INF;
    const Sphere* hit_sphere = nullptr;

    // Find the closest sphere and hit the sphere or not
    for (const Sphere& sphere : spheres) {
        float t = 0;
        if (intersect(ray, sphere, t) && t < t_min) {
            t_min = t;
            hit_sphere = &sphere;
        }
    }

    //////////////////////////////////////////////////////////////////////
    // If there's an intersection, calculate the color at the intersection point
	
    if (hit_sphere) {
        glm::vec3 intersection_point = ray.origin + ray.direction * t_min;
        glm::vec3 normal = glm::normalize(intersection_point - hit_sphere->Position);
        // Calculate color using the Blinn-Phong shading model
		// Blinn_Phong(const glm::vec3& normal, const glm::vec3& viewDir, const glm::vec3& lightDir, const Material& material, const Light& light)
        color = Blinn_Phong(intersection_point, normal, hit_sphere->materials, light, ray.origin);
        
		
		// Check if the intersection point is in shadow
		// bool is_in_shadow(glm::vec3& point, Light& light, const std::vector<Sphere>& spheres, const Sphere& ignore_sphere) {
        //if(is_in_shadow(intersection_point + normal * 1e-4f, light, spheres, *hit_sphere)) {
		if(is_in_shadow(intersection_point + normal * 1e-4f, light, spheres, *hit_sphere)) {
    		color *= 0.2; // Darken the color if in shadow, for demonstration purposes
		}
	}
	//glm::vec3 color;
    return color;
}



void task1_2(int width, int height, string filename){
	// light
	// Light(const glm::vec3 &position, float &intensity) : Position(position), Intensity(intensity){}
	Light light(glm::vec3(-2.0, 1.0, 1.0), 1.0f);

	// materials 
	Material Red_M(glm::vec3(1.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f);
	Material Green_M(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f), 100.0f);
	Material Blue_M(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f), 100.0f);
	
	// ball set
	vector<Sphere> sphere_set;
	// red ball
	sphere_set.push_back( Sphere(glm::vec3(-0.5, -1.0, 1.0),    glm::vec3(1.0, 1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), Red_M)     );
	// green ball 
	sphere_set.push_back( Sphere(glm::vec3(0.5f, -1.0f, -1.0f), glm::vec3(1.0, 1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), Green_M)   );
	// blue ball
	sphere_set.push_back( Sphere(glm::vec3(0.0f, 1.0f, 0.0f),   glm::vec3(1.0, 1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), Blue_M)    );

	// image 
	auto image = make_shared<Image>(width, height);
	// camera
	// Camera(glm::vec3 position, glm::vec3 rotation, float view, float aspect_ratio, int image_width, int image_height)
	Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), 45.0f, 1.0f, width, height);

	// traverse each pixel 
	for (int y = 0; y < camera.Image_height; ++y) {
        for (int x = 0; x < camera.Image_width; x++) {

			// generate the ray from each pixel 
			Ray ray = camera.generate_ray(x, y);
			//std::cout << "Ray direction: " << glm::to_string(ray.direction) << std::endl;
			// trace the ray 
            glm::vec3 color = trace_ray(ray, sphere_set, light);
			//std::cout << color.x << " " << color.y << " " << color.z << std::endl;
			/*
			unsigned char r = static_cast<unsigned char>(255 * color.x);
            unsigned char g = static_cast<unsigned char>(255 * color.y);
            unsigned char b = static_cast<unsigned char>(255 * color.z);*/


			color.r = std::clamp(color.r, 0.0f, 1.0f);
    		color.g = std::clamp(color.g, 0.0f, 1.0f);
    		color.b = std::clamp(color.b, 0.0f, 1.0f);
            color = color * 255.0f;
			image->setPixel(x, y, color.r, color.g, color.b);
        }
    }
	image->writeToFile(filename);
	return ;
}
/*


void task3(){
	// camera 
	Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), 45.0f, width, height);
	
	// light 1 and light 2
	Light light1(glm::vec3(1.0f, 2.0f, 2.0f), );
	Light light2(glm::vec3(-1.0f, 2.0f, -1.0f), );

	// material 
	Material Red_M(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f), 100.0f);
    Material Green_M(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f), 100.0f);
    Material plane_material(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.1f, 0.1f, 0.1f), 0.0f);

	// ellipsoid 
	Ellipsoid red_ellipsoid(glm::vec3(0.5f, 0.0f, 0.5f), glm::vec3(0.5f, 0.6f, 0.2f), red_material);
    
	// sphere 
    Sphere green_sphere(glm::vec3(-0.5f, 0.0f, -0.5f), 1.0f, green_material);
    Plane plane(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)); // Plane y = -1

	// image 
	auto image = make_shared<Image>(width, height);
	// traverse each pixel 
	for (int y = 0; y < camera.Image_height; ++y) {
        for (int x = 0; x < camera.Image_width; x++) {

			// generate the ray from each pixel 
			Ray ray = camera.generate_ray(x, y);
			//std::cout << "Ray direction: " << glm::to_string(ray.direction) << std::endl;
			// trace the ray 
            glm::vec3 color = trace_ray_ellipsoid(ray, sphere_set, light);
			//std::cout << color.x << " " << color.y << " " << color.z << std::endl;
			unsigned char r = static_cast<unsigned char>(255 * color.x);
            unsigned char g = static_cast<unsigned char>(255 * color.y);
            unsigned char b = static_cast<unsigned char>(255 * color.z);
			image->setPixel(x, y, r, g, b);
        }
    }
	image->writeToFile(filename);
	return ;
}

/*
glm::vec3 trace_ray_recurvise(const Ray& ray, const std::vector<Sphere>& spheres, const std::vector<Light>& lights, int depth, int max_depth) {
	if (depth >= max_depth) {
        return glm::vec3(0); // Termination condition based on recursion depth
    }

	glm::vec3 color(0);
    // ... [code for finding the closest sphere intersection and computing local color]

    if (hit_sphere) {
        glm::vec3 intersection_point = ray.origin + t_min * ray.direction;
        glm::vec3 normal = glm::normalize(intersection_point - hit_sphere->center);

        // Compute local color
        glm::vec3 local_color = calculate_lighting(...);

        // If the sphere is reflective, compute reflection
        if (hit_sphere->material.reflective) {
            glm::vec3 reflection_direction = reflect(ray.direction, normal);
            glm::vec3 reflection_origin = intersection_point + EPSILON * normal; // Offset to avoid self-intersection
            Ray reflection_ray(reflection_origin, reflection_direction);
            glm::vec3 reflection_color = trace_ray(reflection_ray, spheres, lights, depth + 1, max_depth);

            color += hit_sphere->material.reflectivity * reflection_color;
        }

        color += local_color; // Combine reflection color with local color
    }

    return color;


}

// Reflection function
glm::vec3 reflect(const glm::vec3& I, const glm::vec3& N) {
    return I - 2.0f * glm::dot(I, N) * N;
}



void task4(){
	// light setting 
	Light light1(glm::vec3(-1.0, 2.0, 1.0), 0.5f);
	Light light2(glm::vec3(0.5, -0.5, 0.0), 0.5f);

	// camera
    Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), 45.0f, 1024, 1024);

	// material
    Material red_material(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f), 100.0f);
    Material blue_material(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f), 100.0f);
    Material reflective_material(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.1f, 0.1f, 0.1f), 0.0f); // Assuming perfect mirror for reflective spheres
    Material back_wall_material(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.1f, 0.1f, 0.1f), 0.0f); // Non-reflective floor material
	Material floor_material(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.1f, 0.1f, 0.1f), 0.0f); // Non-reflective floor material

	// sphere
	std::vector<Sphere> spheres = {
		// Red ball
        Sphere(glm::vec3(-0.5f, -0.7f, 0.5f), 0.3f, red_material),
		// Blue ball 
        Sphere(glm::vec3(1.0f, -0.7f, 0.0f), 0.3f, blue_material),
		// reflective sphere 1
        Sphere(glm::vec3(-0.5f, 0.0f, -0.5f), 0.3f, reflective_material), 
		// reflective sphere 2
        Sphere(glm::vec3(1.5f, 0.0f, -1.5f), 0.3f, reflective_material)
    };

	// floor and back wall
	std::vector<Plane> planes = {
		Plane(glm::vec3(0, 1, 0), glm::vec3(0, -1, 0), floor_material),  // Floor: y = -1
		Plane(glm::vec3(0, 0, 1), glm::vec3(0, 0, -3), back_wall_material) // Back wall: z = -3
	};

	// image 
	auto image = make_shared<Image>(width, height);

}


bool rayIntersectsTriangle(const Ray& ray, const Triangle& tri, float& t, float& u, float& v) {
    const float EPSILON = 0.0000001f;
    glm::vec3 vertex0 = tri.vert0;
    glm::vec3 vertex1 = tri.vert1;
    glm::vec3 vertex2 = tri.vert2;
    glm::vec3 edge1, edge2, h, s, q;
    float a, f, det;

    edge1 = vertex1 - vertex0;
    edge2 = vertex2 - vertex0;
    h = glm::cross(ray.direction, edge2);
    a = glm::dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) {
        return false;    // This ray is parallel to this triangle.
    }

    f = 1.0f / a;
    s = ray.origin - vertex0;
    u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    q = glm::cross(s, edge1);
    v = f * glm::dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    // At this stage we can compute t to find out where the intersection point is on the line.
    t = f * glm::dot(edge2, q);

    if (t > EPSILON) { // ray intersection
        return true;
    } else { // This means that there is a line intersection but not a ray intersection.
        return false;
    }
}



void task_5(){
	// For each pixel in the image
	for each pixel at (x, y) in the image {
		// Generate a ray from the camera through the pixel
		Ray ray = camera.generate_ray(x, y);

		// First test against the bounding volume
		if (rayIntersectsBoundingVolume(ray, boundingVolume)) {
			float nearestDistance = INF;
			Triangle nearestTriangle;

			// Then test against each triangle in the mesh
			for each triangle in mesh {
				float distance;
				if (rayIntersectsTriangle(ray, triangle, distance)) {
					if (distance < nearestDistance) {
						nearestDistance = distance;
						nearestTriangle = triangle;
					}
				}
			}

			// If an intersection was found, shade the pixel
			if (nearestDistance < INF) {
				glm::vec3 color = shade(nearestTriangle, material, light, ray);
				image.setPixel(x, y, color);
			}
		}
	}
	return ;
}

*/