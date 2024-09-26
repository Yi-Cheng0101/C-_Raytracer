

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