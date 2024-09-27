
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