#pragma once
#ifndef LIGHT_H_
#define LIGHT_H_

#include <glm/glm.hpp>

class Light{
	public:
		glm::vec3 Position;
		glm::vec3 color;
		float Intensity;
	
		Light(const glm::vec3 &position, float intensity) : Position(position), Intensity(intensity){
			color = glm::vec3(1.0, 1.0, 1.0);
		}
};

#endif 