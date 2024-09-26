#define GLM_ENABLE_EXPERIMENTAL
#include <cassert>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/matrix_inverse.hpp>    
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

class Hit{
	public:
		glm::vec3 Normal;
		glm::vec3 Position;
		float T;
		Hit(const glm::vec3 &position, const glm::vec3 &normal, float t) : Position(position), Normal(normal), T(t){}
};
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

class Material{
	public:
		glm::vec3 Diffuse;
		glm::vec3 Specular;
		glm::vec3 Ambient;
		bool Reflective;
		float Exponent;
		Material(const glm::vec3& diffuse, const glm::vec3& specular, const glm::vec3& ambient, float exponent, bool reflective): Diffuse(diffuse), Specular(specular), Ambient(ambient), Exponent(exponent), Reflective(reflective){}
};

class Model {
	public: 
		glm::vec3 Position;
		glm::vec3 Rotation;
		glm::vec3 Scale;
		Material material;

		Model(const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale, const Material& material)
        : Position(position), Rotation(rotation), Scale(scale), material(material) {}
   		virtual bool intersect(const Ray& ray, float& t, glm::vec3& hit_normal) const = 0;
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
    
	Ray generate_ray_task8(int x, int y) const {
		float halfHeight = tan(View / 2);
		float halfWidth = Aspect_ratio * halfHeight;

		float pixelWidth = (halfWidth * 2) / Image_width;
		float pixelHeight = (halfHeight * 2) / Image_height;

		float base = -tan(View/2);
		float u = x * pixelWidth  + base + pixelWidth * 0.5f;
		float v = y * pixelHeight + base + pixelHeight * 0.5f;
		
		glm::vec3 rayDir = glm::normalize(glm::vec3(-2.0,v, u) - Position);
		glm::vec3 rayPos = Position;
		return Ray(rayPos, rayDir);
	}

	Ray generate_ray(int x, int y) const {
		float halfHeight = tan(View / 2);
		float halfWidth = Aspect_ratio * halfHeight;

		float pixelWidth = (halfWidth * 2) / Image_width;
		float pixelHeight = (halfHeight * 2) / Image_height;

		float base = -tan(View/2);
		float u = x * pixelWidth  + base + pixelWidth * 0.5f;
		float v = y * pixelHeight + base + pixelHeight * 0.5f;
		
		glm::vec3 rayDir = glm::normalize(glm::vec3(u ,v, 4.0f) - Position);
		glm::vec3 rayPos = Position;
		return Ray(rayPos, rayDir);
	}
};


class Sphere : public Model{
	public:
		float radius;
		Sphere(const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale, const Material& material)
		: Model(position, rotation, scale, material){
			radius = scale.x;
		};

		bool intersect(const Ray& ray, float& t, glm::vec3& hit_normal) const override {
			glm::vec3 oc = ray.origin - Position;
			float a = glm::dot(ray.direction, ray.direction);
			float b = 2.0f * glm::dot(oc, ray.direction);
			float c = glm::dot(oc, oc) - radius * radius;
			float discriminant = b * b - 4 * a * c;

			if (discriminant < 0) {
				return false;
			} else {
				discriminant = std::sqrt(discriminant);
				float t0 = (-b - discriminant) / (2 * a);
				float t1 = (-b + discriminant) / (2 * a);

				if (t0 > t1) std::swap(t0, t1);
				if (t0 < 0) {
					t0 = t1; // If t0 is negative, let's use t1 instead
					if (t0 < 0) return false; // Both t0 and t1 are negative
				}

				t = t0;
				hit_normal = glm::normalize((ray.origin + ray.direction * t) - Position);
				return true;
			}
    }
};

class Move_Sphere : public Model {
public:
    glm::vec3 center;
    float radius;
    glm::vec3 velocity;  // Velocity for motion

    Move_Sphere(const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale, const Material& material, const glm::vec3& vel)
        : Model(position, rotation, scale, material) {
			radius = scale.x;
			velocity = glm::vec3(0.0f);
	};
	
    void updatePosition(float time) {
        center += velocity * time;  // Update center based on velocity and time
    }

	//bool intersect(const Ray& ray, float& t_near, glm::vec3& hit_normal) override {
		bool intersect(const Ray& ray, float& t, glm::vec3& hit_normal) const override {
			glm::vec3 oc = ray.origin - Position;
			float a = glm::dot(ray.direction, ray.direction);
			float b = 2.0f * glm::dot(oc, ray.direction);
			float c = glm::dot(oc, oc) - radius * radius;
			float discriminant = b * b - 4 * a * c;

			if (discriminant < 0) {
				return false;
			} else {
				discriminant = std::sqrt(discriminant);
				float t0 = (-b - discriminant) / (2 * a);
				float t1 = (-b + discriminant) / (2 * a);

				if (t0 > t1) std::swap(t0, t1);
				if (t0 < 0) {
					t0 = t1; // If t0 is negative, let's use t1 instead
					if (t0 < 0) return false; // Both t0 and t1 are negative
				}

				t = t0;
				hit_normal = glm::normalize((ray.origin + ray.direction * t) - Position);
				return true;
			}
    }
};


class Ellipsoid : public Model {
public:
    glm::mat4 ellipsoid_E;

    Ellipsoid(const glm::vec3 &position, const glm::vec3 &rotation, const glm::vec3 &scale, 
              const Material& material)
        : Model(position, rotation, scale, material) {
        ellipsoid_E = glm::translate(glm::mat4(1.0), position) *
                      glm::scale(glm::mat4(1.0), scale);
    }
	

    bool intersect(const Ray& ray, float& t_near, glm::vec3& hit_normal) const override {
        // Transform the ray to object space
        glm::mat4 inverse_E = glm::inverse(ellipsoid_E);
        glm::vec3 ray_origin_obj = glm::vec3(inverse_E * glm::vec4(ray.origin, 1.0));
        glm::vec3 ray_direction_obj = glm::normalize(glm::vec3(inverse_E * glm::vec4(ray.direction, 0.0)));

        // Compute the quadratic formula coefficients
        float a = glm::dot(ray_direction_obj, ray_direction_obj);
        float b = 2.0 * glm::dot(ray_origin_obj, ray_direction_obj);
        float c = glm::dot(ray_origin_obj, ray_origin_obj) - 1.0;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) {
            return false; // No intersection
        }

        float sqrtD = glm::sqrt(discriminant);
        float t1 = (-b - sqrtD) / (2 * a);
        float t2 = (-b + sqrtD) / (2 * a);

        if (t1 > t2) swap(t1, t2);
        if (t1 < 0) {
            t1 = t2; // If t1 is negative, use t2 instead
            if (t1 < 0) return false; // Both are negative
        }

        t_near = t1;

        glm::vec3 intersection_point_obj = ray_origin_obj + t_near * ray_direction_obj;
		glm::vec3 actual_intersection = glm::vec3(ellipsoid_E * glm::vec4(intersection_point_obj, 1.0));
		glm::vec3 diff = actual_intersection - ray.origin;
		float t_acutal = glm::length(diff);
		t_near = t_acutal;
        hit_normal = glm::normalize(glm::vec3(glm::transpose(inverse_E) * glm::vec4(intersection_point_obj, 0.0)));
        
        return true; 
    }
};


class Plane : public Model{
	public:
		glm::vec3 Normal = glm::vec3(0, 0, 0);
		Plane(const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale, const Material& material)
		: Model(position, rotation, scale, material){
			Normal = rotation;
		}

		// Implement the ray-plane intersection method
		bool intersect(const Ray& ray, float& t, glm::vec3& hit_normal) const override {
			float denominator = glm::dot(this->Normal, ray.direction);
			
			if (std::abs(denominator) > 1e-6) {  // Check if the ray is parallel to the plane
				glm::vec3 p0_to_origin = Position - ray.origin;
				t = glm::dot(p0_to_origin, this->Normal) / denominator;
				
				if (t >= 0) {  // Only consider intersections in front of the ray origin
					hit_normal = this->Normal;  // Set the hit normal
					return true;
				}
			}
			return false;
	}
};

//  Ref: "Fundamentals of Computer Graphics" by Peter Shirley
class Box : public Model {
public:
    glm::vec3 halfSizes; 

    Box(const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale, const Material& material)
        : Model(position, rotation, scale, material), halfSizes(scale) {}

    virtual bool intersect(const Ray& ray, float& t, glm::vec3& hit_normal) const override {
        // Implement the intersection logic here
        glm::vec3 tMin = (Position - halfSizes - ray.origin) / ray.direction;
        glm::vec3 tMax = (Position + halfSizes - ray.origin) / ray.direction;
        
        glm::vec3 t1 = glm::min(tMin, tMax);
        glm::vec3 t2 = glm::max(tMin, tMax);

        float tNear = std::max(std::max(t1.x, t1.y), t1.z);
        float tFar = std::min(std::min(t2.x, t2.y), t2.z);

        if (tNear > tFar || tFar < 0.0f) {
            return false; // No intersection
        }

        t = tNear;

        // Determine the hit normal
        if (tNear == t1.x) {
            hit_normal = glm::vec3((tMin.x > tMax.x) ? 1.0f : -1.0f, 0.0f, 0.0f);
        } else if (tNear == t1.y) {
            hit_normal = glm::vec3(0.0f, (tMin.y > tMax.y) ? 1.0f : -1.0f, 0.0f);
        } else {
            hit_normal = glm::vec3(0.0f, 0.0f, (tMin.z > tMax.z) ? 1.0f : -1.0f);
        }

        return true; // There is an intersection
    }
};


class Vertice{
	public:
		glm::vec3 Pos;
		glm::vec3 Nor;
		Vertice(glm::vec3 pos, glm::vec3 nor) : Pos(pos), Nor(nor){}
};


class Triangle {
public:
    Vertice t0, t1, t2;  
    float w, u, v;

    // Default constructor
    Triangle() : t0(Vertice(glm::vec3(0), glm::vec3(0))),
                 t1(Vertice(glm::vec3(0), glm::vec3(0))),
                 t2(Vertice(glm::vec3(0), glm::vec3(0))),
                 w(0), u(0), v(0) {}

    // Constructor with vertices
    Triangle(const Vertice &t0_, const Vertice &t1_, const Vertice &t2_)
        : t0(t0_), t1(t1_), t2(t2_), w(0), u(0), v(0) {}

    // Coordinate setting function
    void coord(float w_, float u_, float v_) {
        w = w_; 
        u = u_;
        v = v_;
    }
};


class IntersectionResult {
    public:
        float distance;
        float u, v, w;  // Barycentric coordinates
        glm::vec3 position;  // Intersection position
        glm::vec3 normal;    // Normal at the intersection
        bool hit;            // Flag to indicate if there was an intersection
        IntersectionResult() : distance(0.0f), u(0.0f), v(0.0f), w(0.0f), position(glm::vec3(0.0f)), normal(glm::vec3(0.0f)), hit(false) {}
};



class BoundingSphere {
public:
    glm::vec3 center;
    float radius;

    BoundingSphere(const glm::vec3& center, float radius)
        : center(center), radius(radius) {}
};


// function
void task0(int width, int height, string filename);
void task1_2(int width, int height, string filename);
void task3(int width, int height, string filename);
void task4(int width, int height, string filename);
void task5(int width, int height, string filename);
void task8(int width, int height, string filename);
void task9(int width, int height, string filename);

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

	switch(scene) {
		case 0:
			// multi thread with box 
            task0(image_size, image_size, file_name);
            break;
        case 1:
            task1_2(image_size, image_size, file_name);
            break;
        case 2:
            task1_2(image_size, image_size, file_name);
			break;
		case 3:
			task3(image_size, image_size, file_name);
			break;
		case 4:
			task4(image_size, image_size, file_name);
			break;
		case 5:
			task4(image_size, image_size, file_name);
			break;
		// bonus for antil
		case 6:
			task5(image_size, image_size, file_name);
			break;
		case 7:
			task5(image_size, image_size, file_name);
			break;
		case 8:
			task8(image_size, image_size, file_name);
			break;
		case 9:
			// motion blur with antialiasing
			task9(image_size, image_size, file_name);
			break;
        default:
            break;
    }

	return 0;
}



glm::vec3 Blinn_Phong(const glm::vec3& intersection_point, const glm::vec3& normal, const Material& material, const Light& light, const glm::vec3& view_position) {
    glm::vec3 lightDir = glm::normalize(light.Position - intersection_point);
    glm::vec3 viewDir = glm::normalize(view_position - intersection_point);

    // Diffuse reflection
    float diffuseFactor = std::max(glm::dot(normal, lightDir), 0.0f);
    glm::vec3 diffuse = material.Diffuse * light.color * diffuseFactor * light.Intensity;

    // Specular reflection
    glm::vec3 halfwayDir = glm::normalize(lightDir + viewDir);
    float specularFactor = std::pow(std::max(glm::dot(normal, halfwayDir), 0.0f), material.Exponent);
    glm::vec3 specular = material.Specular * light.color * specularFactor * light.Intensity;

    return diffuse + specular;
}


glm::vec3 traceRayRecursive_new(const float &t0, const float &t1, const Ray& ray, const std::vector<std::unique_ptr<Model>>& models, const std::vector<Light>& lights, int depth) {


    glm::vec3 color(0.0); // Background color
    float t_near = INF;
    const Model* hitObject = nullptr;
    glm::vec3 hitNormal;

    // Find the intersection
    for (const auto& model : models) {
        float t = INF;
        glm::vec3 normal;
        if (model->intersect(ray, t, normal) && t < t_near && t > t0 && t1 > t) {
            t_near = t;
            hitObject = model.get();
            hitNormal = normal;
        }
    }

    if (hitObject) {
		glm::vec3 intersection_point = ray.origin + ray.direction * t_near;      // hit point 
        glm::vec3 view_position = ray.origin;									// Assuming the view position is the ray origin
		color = hitObject->material.Ambient;

		// reflective model
		
        if(hitObject->material.Reflective &&  0 < depth){
            glm::vec3 refDir = glm::reflect(ray.direction, hitNormal);
            Ray refray(intersection_point, refDir);
			return color + traceRayRecursive_new(0.001, INF, refray, models, lights, depth-1);
    	}
		// multiple lights
		for(const auto& light : lights) {
			glm::vec3 lightDir =  glm::normalize(light.Position - intersection_point);
            float lightDist = glm::distance(light.Position, intersection_point);
            Ray sray(intersection_point, lightDir);
			
			float t_near_ = INF;
			const Model* hitObject_ = nullptr;
			glm::vec3 hitNormal_;
			for (const auto& model : models) {
				float t_ = INF;
				glm::vec3 normal_;
				if (model->intersect(sray, t_, normal_) && t_ < t_near_ && t_ > 0.001f && t_ < lightDist) {
					t_near_ = t_;
            		hitObject_ = model.get();
            		hitNormal_ = normal_;
				}
			}
            if(!hitObject_){ // no hits
                color += Blinn_Phong(intersection_point, hitNormal, hitObject->material, light, view_position);
            }
		}
    }
    return color;
}

void task1_2(int width, int height, string filename){
	
	// light	
	std::vector<Light> lights;
    lights.push_back(Light(glm::vec3(-2.0, 1.0, 1.0), 1.0f));
	
	// material 
	Material Red_M(glm::vec3(1.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);
    Material Green_M(glm::vec3(0.0, 1.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);
    Material Blue_M(glm::vec3(0.0, 0.0, 1.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);

	// models
    std::vector<std::unique_ptr<Model>> models;
    models.push_back(std::make_unique<Sphere>(glm::vec3(-0.5, -1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Red_M));
    models.push_back(std::make_unique<Sphere>(glm::vec3(0.5, -1.0, -1.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Green_M));
    models.push_back(std::make_unique<Sphere>(glm::vec3(0.0, 1.0, 0.0),   glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Blue_M));

	// image 
	auto image = make_shared<Image>(width, height);
	
	// camera
	Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), 45.0f, 1.0f, width, height);

	// traverse each pixel 
	for (int y = 0; y < camera.Image_height; ++y) {
        for (int x = 0; x < camera.Image_width; x++) {
			// generate the ray from each pixel 
			Ray ray = camera.generate_ray(x, y);
			glm::vec3 color = traceRayRecursive_new(0.0f, INF, ray, models, lights, 1000);
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


void task3(int width, int height, string filename){
	// camera 
	Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), 45.0f, 1.0f, width, height);

	// light 1 and light 2
	vector<Light> lights{
        Light(glm::vec3(1.0, 2.0, 2.0), 0.5f),
        Light(glm::vec3(-1.0, 2.0, -1.0), 0.5f)
    };

	// material 
    Material redMaterial(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f), 100.0f, false);
    Material greenMaterial(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f), 100.0f, false);
    Material planeMaterial(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.1f, 0.1f, 0.1f), 0.0f, false);

	// scene 
	std::vector<std::unique_ptr<Model>> models;
    models.push_back(std::make_unique<Sphere>(glm::vec3(-0.5f, 0.0f, -0.5f), glm::vec3(1.0, 1.0, 1.0), glm::vec3(1.0, 1.0, 1.0), greenMaterial));
    models.push_back(std::make_unique<Ellipsoid>(glm::vec3(0.5f, 0.0f, 0.5f), glm::vec3(0.5f, 0.6f, 0.2f), glm::vec3(0.5, 0.6, 0.2), redMaterial));
    models.push_back(std::make_unique<Plane>(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0, 1.0, 1.0), planeMaterial));

	// image 
	auto image = make_shared<Image>(width, height);   

    // Render the scene
    for (int y = 0; y < camera.Image_height; ++y) {
        for (int x = 0; x < camera.Image_width; ++x) {
            Ray ray = camera.generate_ray(x, y);
			glm::vec3 color = traceRayRecursive_new(0.0f, INF, ray, models, lights, 1000);
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

void task4(int width, int height, string filename){
	// camera 
	Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), 45.0f, 1.0f, width, height);

	// light 1 and light 2
	vector<Light> lights{
        Light(glm::vec3(-1.0, 2.0, 1.0), 0.5f),
        Light(glm::vec3(0.5, -0.5, 0.0), 0.5f)
    };

	// scene 
	std::vector<std::unique_ptr<Model>> models;

	// material 
   	Material redMaterial(glm::vec3(1.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);
    Material blueMaterial(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 0.5f), glm::vec3(0.1f, 0.1f, 0.1f), 100.0f, false);
	Material R_Material(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), 100.0f, true);
	Material floorMaterial(glm::vec3(1.0, 1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.1, 0.1, 0.1), 0.0f, false);                   
	Material wallMaterial(glm::vec3(1.0, 1.0, 1.0),  glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.1, 0.1, 0.1), 0.0f, false);                   
	
	Material boxMaterial(glm::vec3(0.1, 0.2, 0.1), glm::vec3(0.0, 0.8, 0.0), glm::vec3(0.1, 0.1, 0.1), 32.0f, false);

	// sphere
    models.push_back(std::make_unique<Sphere>(glm::vec3(0.5, -0.7, 0.5), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.3, 0.3, 0.3), redMaterial));
    models.push_back(std::make_unique<Sphere>(glm::vec3(1.0, -0.7, 0.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.3, 0.3, 0.3), blueMaterial));
	models.push_back(std::make_unique<Sphere>(glm::vec3(-0.5, 0.0, -0.5), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0, 1.0, 1.0), R_Material));
	models.push_back(std::make_unique<Sphere>(glm::vec3(1.5, 0.0, -1.5), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0, 1.0, 1.0),  R_Material));

	// Floor
	models.push_back(std::make_unique<Plane>(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0, 1.0, 1.0), floorMaterial));

	// Wall
	models.push_back(std::make_unique<Plane>(glm::vec3(0.0f, 0.0f, -3.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0, 1.0, 1.0), wallMaterial));

	// image 
	auto image = make_shared<Image>(width, height);   

    // Render the scene
    for (int y = 0; y < camera.Image_height; ++y) {
        for (int x = 0; x < camera.Image_width; ++x) {
            Ray ray = camera.generate_ray(x, y);
			glm::vec3 color = traceRayRecursive_new(0.0f, INF, ray, models, lights, 1000);
			color.r = std::clamp(color.r, 0.0f, 1.0f);
    		color.g = std::clamp(color.g, 0.0f, 1.0f);
    		color.b = std::clamp(color.b, 0.0f, 1.0f);
			color = color * 255.0f;
            image->setPixel(x, y, color.r , color.g, color.b);
        }
    }
	image->writeToFile(filename);
	return ;
}

///////////////////////////////////////////////////////////////////////////

bool rayIntersectsBoundingVolume(const Ray& ray, const BoundingSphere& sphere) {
    glm::vec3 oc = ray.origin - sphere.center; // Vector from ray origin to sphere center
    float b = glm::dot(oc, ray.direction); // This is 'b' in the quadratic equation
    float c = glm::dot(oc, oc) - sphere.radius * sphere.radius; 
    float discriminant = b * b - c; 

    // A negative discriminant corresponds to ray missing the sphere
    if (discriminant < 0.0f) {
        return false;
    }

    // Check if the intersection is in the positive direction of the ray
    float sqrtd = sqrt(discriminant);
    float t0 = -b - sqrtd; // First possible intersection t
    float t1 = -b + sqrtd; // Second possible intersection t

    // If both t0 and t1 are negative, the sphere is behind the ray origin
    if (t0 < 0.0f && t1 < 0.0f) {
        return false;
    }

    // If t0 is negative, then the intersection we want is t1
    if (t0 < 0.0f) {
        return true;
    }

    // If we reach here, t0 is the first intersection in the positive direction
    return true;
}


// triangle intersction function 
bool rayIntersectsTriangle(const Ray& ray, const Triangle& triangle, IntersectionResult& result) {
    const float EPSILON = 0.0000001f;
    glm::vec3 vertex0 = triangle.t0.Pos;
    glm::vec3 vertex1 = triangle.t1.Pos;
    glm::vec3 vertex2 = triangle.t2.Pos;

    glm::vec3 edge1 = vertex1 - vertex0;
    glm::vec3 edge2 = vertex2 - vertex0;
    glm::vec3 h = glm::cross(ray.direction, edge2);
    float a = glm::dot(edge1, h);

    if (a > -EPSILON && a < EPSILON)
        return false;    // Ray is parallel to this triangle.

    float f = 1.0f / a;
    glm::vec3 s = ray.origin - vertex0;
    float u = f * glm::dot(s, h);

    if (u < 0.0 || u > 1.0)
        return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.direction, q);

    if (v < 0.0 || u + v > 1.0)
        return false;

    float t = f * glm::dot(edge2, q);
    if (t > EPSILON) { // Ray intersection
        result.distance = t;
        result.u = u;
        result.v = v;
        result.w = 1.0f - u - v;
        result.hit = true;
		//std::cout << result.w << " " << result.v << " " << result.u << std::endl;

        return true;
    } else {
        return false;
    }
}

glm::vec3 shade(
    const Triangle& triangle, 
    const Material& material, 
    const Light& light, 
    const glm::vec3& intersectionPoint, 
    const glm::vec3& cameraPosition,
	IntersectionResult& result
) {
    // Calculate the normal at the intersection point using barycentric coordinates
	glm::vec3 normal(0.0f);
	normal.x = result.w * triangle.t0.Nor.x +  triangle.t1.Nor.x * result.u + triangle.t2.Nor.x * result.v;
	normal.y = result.w * triangle.t0.Nor.y +  triangle.t1.Nor.y * result.u + triangle.t2.Nor.y * result.v;
	normal.z = result.w * triangle.t0.Nor.z +  triangle.t1.Nor.z * result.u + triangle.t2.Nor.z * result.v;
	normal.x = std::clamp(normal.x, -1.0f, 1.0f);
    normal.y = std::clamp(normal.y, -1.0f, 1.0f);
    normal.z = std::clamp(normal.z, -1.0f, 1.0f);
	normal = glm::normalize(normal);


	// Light direction from the point to the light
    glm::vec3 lightDir = glm::normalize(light.Position - intersectionPoint);

    // View direction from the point to the camera
    glm::vec3 viewDir = glm::normalize(cameraPosition - intersectionPoint);

    // Calculate the half vector between the light direction and the view direction
    glm::vec3 halfVector = glm::normalize(lightDir + viewDir);

    // Ambient component is a constant set by the material
    glm::vec3 ambient = material.Ambient * light.Intensity;

    // Diffuse component depends on the angle between the light direction and the normal
    float diff = glm::max(glm::dot(normal, lightDir), 0.0f);
    glm::vec3 diffuse = material.Diffuse * diff * light.Intensity;

    // Specular component depends on the angle between the half vector and the normal
    float spec = glm::pow(glm::max(glm::dot(normal, halfVector), 0.0f), material.Exponent);
    glm::vec3 specular = material.Specular * spec * light.Intensity;

    // Combine the components
    glm::vec3 result_color = ambient + diffuse + specular;
    
    // Ensure that the color components are clamped between 0.0 and 1.0
    result_color = glm::clamp(result_color, 0.0f, 1.0f);

    return result_color;
}

BoundingSphere calculateBoundingVolumeForMesh(const std::vector<float>& posBuf) {
    // Assuming posBuf is a flat list of vertices (x,y,z)
    glm::vec3 min(std::numeric_limits<float>::max());
    glm::vec3 max(std::numeric_limits<float>::lowest());

    for (size_t i = 0; i < posBuf.size(); i += 3) {
        glm::vec3 vertex(posBuf[i], posBuf[i + 1], posBuf[i + 2]);
        min = glm::min(min, vertex);
        max = glm::max(max, vertex);
    }

    glm::vec3 center = (min + max) * 0.5f;
    float radius = glm::distance(center, max);

    return BoundingSphere(center, radius);
}


void task5(int width, int height, string filename){
	glm::mat4 E = glm::mat4(1.0); // load identity

	// camera
	Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), 45.0f, 1.0f, width, height);

	// light
	vector<Light> light{
		Light(glm::vec3(-1.0, 1.0, 1.0), 1.0f)
    };
	

	// Materials
   	Material B_Material(glm::vec3(0.0, 0.0, 1.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);             
	
	// load model
	vector<float> posBuf; // list of vertex positions
	vector<float> norBuf; // list of vertex normals
	vector<float> texBuf; // list of vertex texture coords
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	string meshName = "../resources/bunny.obj";
	std::string errStr;
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
			cout << "Number of vertices: " << posBuf.size()/3 << endl;
		}
	}

	// mesh setting 
	vector<Triangle> meshTriangles;
	vector<Vertice> vertice;
	// mesh setting 
	for(int i=0; i < posBuf.size(); i+=3){
		Vertice temp(glm::vec3(posBuf[i], posBuf[i+1], posBuf[i+2]), glm::vec3(norBuf[i], norBuf[i+1], norBuf[i+2]));
		vertice.push_back(temp);
	}
	
	for(int i=0; i < vertice.size(); i+=3){
		meshTriangles.push_back(Triangle(vertice[i], vertice[i+1], vertice[i+2]));
	}

	BoundingSphere boundingVolume = calculateBoundingVolumeForMesh(posBuf);
	// image 
	auto image = make_shared<Image>(width, height);   

    // Render the scene
    for (int y = 0; y < camera.Image_height; ++y) {
        for (int x = 0; x < camera.Image_width; ++x) {
			glm::vec3 color(0.0f, 0.0f, 0.0f);
            Ray ray = camera.generate_ray(x, y);
			glm::vec3 intersectionPoint;

			// First test against the bounding volume
			if (rayIntersectsBoundingVolume(ray, boundingVolume)) {
				float nearestDistance = std::numeric_limits<float>::infinity();
				Triangle nearestTriangle;
				IntersectionResult intersection;
				// Then test against each triangle in the mesh
				for (const auto& triangle : meshTriangles) { 
					//IntersectionResult intersection;
					if (rayIntersectsTriangle(ray, triangle, intersection) && intersection.hit) {
						if (intersection.distance < nearestDistance) {
							nearestDistance = intersection.distance;
							nearestTriangle = triangle;
							intersectionPoint = ray.origin + ray.direction * intersection.distance;
						}
					}
				}
				// If an intersection was found, shade the pixel
				if (nearestDistance < std::numeric_limits<float>::infinity()) {
					color = shade(nearestTriangle, B_Material, light[0], intersectionPoint, camera.Position, intersection);
					
					float w = 1.0f - nearestTriangle.u - nearestTriangle.v;
					glm::vec3 normal(0.0f);
				}
			}

			color.r = std::clamp(color.r, 0.0f, 1.0f);
    		color.g = std::clamp(color.g, 0.0f, 1.0f);
    		color.b = std::clamp(color.b, 0.0f, 1.0f);
			image->setPixel(x, y, color.r * 255, color.g *255, color.b *255);
        }
    }
	image->writeToFile(filename);

	return ;
}


void task8(int width, int height, string filename){
	
	std::vector<Light> lights;
    lights.push_back(Light(glm::vec3(-2.0, 1.0, 1.0), 1.0f));
	
	
	Material Red_M(glm::vec3(1.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);
    Material Green_M(glm::vec3(0.0, 1.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);
    Material Blue_M(glm::vec3(0.0, 0.0, 1.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);

    std::vector<std::unique_ptr<Model>> models;
	// pos rotate scale material 
    models.push_back(std::make_unique<Sphere>(glm::vec3(-0.5, -1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Red_M));
    models.push_back(std::make_unique<Sphere>(glm::vec3(0.5, -1.0, -1.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Green_M));
    models.push_back(std::make_unique<Sphere>(glm::vec3(0.0, 1.0, 0.0),   glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Blue_M));
	
	// image 
	auto image = make_shared<Image>(width, height);
	
	// camera
	Camera camera(glm::vec3(-3.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), 60.0f, 1.0f, width, height);

	// traverse each pixel 
	for (int y = 0; y < camera.Image_height; ++y) {
        for (int x = 0; x < camera.Image_width; x++) {

			// generate the ray from each pixel 
			Ray ray = camera.generate_ray_task8(x, y);
			// trace the ray 
            //glm::vec3 color = trace_ray(ray, models, lights);
			glm::vec3 color = traceRayRecursive_new(0, INF, ray, models, lights, 5);
			//glm::vec3 color = traceRayRecursive(ray, models, lights, 5);
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

// Multithreading - process the image pixels in parallel.
void render_section(int start, int end, int width, const Camera& camera, Image* image, const std::vector<std::unique_ptr<Model>>& models, const std::vector<Light>& lights) {
    for (int y = start; y < end; ++y) {
        for (int x = 0; x < width; ++x) {
            float sample_x = x + 0.5f;  // Sample at the pixel center
            float sample_y = y + 0.5f;
            Ray ray = camera.generate_ray(sample_x, sample_y);
            glm::vec3 color = traceRayRecursive_new(0, INF, ray, models, lights, 5);

            color = glm::clamp(color, 0.0f, 1.0f);
            color *= 255.0f;
            image->setPixel(x, y, color.r, color.g, color.b);
        }
    }
}


void task0(int width, int height, std::string filename){
	// get the current thread 
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);
    Image image(width, height);
	Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), 45.0f, 1.0f, width, height);
    std::vector<Light> lights;
    lights.push_back(Light(glm::vec3(-2.0, 1.0, 1.0), 1.0f));
	
	
	Material Red_M(glm::vec3(1.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);
    Material Green_M(glm::vec3(0.0, 1.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);
    Material Blue_M(glm::vec3(0.0, 0.0, 1.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);

    std::vector<std::unique_ptr<Model>> models;
	// pos rotate scale material 

	// box
	Material boxMaterial(glm::vec3(0.1, 0.2, 0.1), glm::vec3(0.0, 0.8, 0.0), glm::vec3(0.1, 0.1, 0.1), 32.0f, false);
    models.push_back(std::make_unique<Box>(glm::vec3(1.0, 0.7, 0.5), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.2, 0.2, 0.2), boxMaterial));
	models.push_back(std::make_unique<Box>(glm::vec3(1.0, 0.3, 1.7), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.2, 0.2, 0.2), boxMaterial));
	models.push_back(std::make_unique<Box>(glm::vec3(0.5, 0.5, -0.9), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.2, 0.2, 0.2), boxMaterial));

    models.push_back(std::make_unique<Sphere>(glm::vec3(-0.5, -1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Red_M));
    models.push_back(std::make_unique<Sphere>(glm::vec3(0.5, -1.0, -1.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Green_M));
    models.push_back(std::make_unique<Sphere>(glm::vec3(0.0, 1.0, 0.0),   glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Blue_M));

    int rows_per_thread = height / num_threads;
    
    for (int i = 0; i < num_threads; ++i) {
        int start_y = i * rows_per_thread;
        int end_y = (i + 1) == num_threads ? height : start_y + rows_per_thread;
        threads[i] = std::thread(render_section, start_y, end_y, width, std::ref(camera), &image, std::ref(models), std::ref(lights));
    }

    for (auto& t : threads) {
        t.join();
    }

    image.writeToFile(filename);
}


glm::vec3 interpolatePosition(const glm::vec3& start, const glm::vec3& end, float time) {
    return start + time * (end - start);
}

void task9(int width, int height, string filename){
    std::vector<Light> lights;
    lights.push_back(Light(glm::vec3(-2.0, 1.0, 1.0), 1.0f));

    Material movingMaterial(glm::vec3(1.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);
    std::vector<std::unique_ptr<Model>> models;
	Material Red_M(glm::vec3(1.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);
    Material Green_M(glm::vec3(0.0, 1.0, 0.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);
    Material Blue_M(glm::vec3(0.0, 0.0, 1.0), glm::vec3(1.0, 1.0, 0.5), glm::vec3(0.1, 0.1, 0.1), 100.0f, false);

    // Static models
    models.push_back(std::make_unique<Sphere>(glm::vec3(-0.5, -1.0, 1.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), movingMaterial));
	models.push_back(std::make_unique<Sphere>(glm::vec3(0.5, -1.0, -1.0), glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Green_M));
    models.push_back(std::make_unique<Sphere>(glm::vec3(0.0, 1.0, 0.0),   glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0), Blue_M));
	
	Material boxMaterial(glm::vec3(0.1, 0.2, 0.1), glm::vec3(0.0, 0.8, 0.0), glm::vec3(0.1, 0.1, 0.1), 32.0f, false);
    models.push_back(std::make_unique<Box>(glm::vec3(1.0, 0.7, 0.5), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.2, 0.2, 0.2), boxMaterial));
	models.push_back(std::make_unique<Box>(glm::vec3(1.0, 0.3, 1.7), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.2, 0.2, 0.2), boxMaterial));
	models.push_back(std::make_unique<Box>(glm::vec3(0.5, 0.5, -0.9), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.2, 0.2, 0.2), boxMaterial));
    auto image = make_shared<Image>(width, height);
    Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), 45.0f, 1.0f, width, height);

    int numFrames = 10;  // Number of sub-frames for motion blur
    float timeStep = 1.0f / numFrames;

    // Process each pixel
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            glm::vec3 accumulatedColor(0.0f, 0.0f, 0.0f);
            for (int frame = 0; frame < numFrames; ++frame) {
                float time = frame * timeStep;
                
                // Update the position of the moving sphere
                glm::vec3 newPos = interpolatePosition(glm::vec3(0.5, -1.0, -1.0), glm::vec3(1.0, 1.0, 1.0), time);
                models[0]->Position = newPos;
                
                Ray ray = camera.generate_ray(x, y);
                glm::vec3 color = traceRayRecursive_new(0.0f, INF, ray, models, lights, 1000);
                accumulatedColor += color;
            }
            // Average the color
            glm::vec3 finalColor = accumulatedColor / static_cast<float>(numFrames);
            finalColor.r = std::clamp(finalColor.r, 0.0f, 1.0f);
            finalColor.g = std::clamp(finalColor.g, 0.0f, 1.0f);
            finalColor.b = std::clamp(finalColor.b, 0.0f, 1.0f);
            finalColor = finalColor * 255.0f;
            image->setPixel(x, y, finalColor.r, finalColor.g, finalColor.b);
        }
    }
    image->writeToFile(filename);
}
