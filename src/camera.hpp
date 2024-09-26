#pragma once
#ifndef _CAMERA_H_
#define _CAMERA_H_

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
