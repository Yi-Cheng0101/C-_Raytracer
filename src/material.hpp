#include "material.hpp"

class Material{
	public:
		glm::vec3 diffuse;
		glm::vec3 specular;
		glm::vec3 ambient;
		bool reflective;
		float exponent;
		Material(const glm::vec3& _diffuse, const glm::vec3& _specular, const glm::vec3& _ambient, float _exponent, bool _reflective): diffuse(_diffuse), specular(_specular), ambient(_ambient), exponent(_exponent), reflective(_reflective){}
};