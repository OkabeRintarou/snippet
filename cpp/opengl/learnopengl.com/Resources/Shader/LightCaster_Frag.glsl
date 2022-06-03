#version 330 core

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

struct Light {
    vec3 direction;
    vec3 position;
    float cut_off;
    float out_cut_off;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;
};

in vec3 frag_pos;
in vec3 normal;
in vec2 tex_coords;

out vec4 color;

uniform vec3 view_pos;
uniform Material material;
uniform Light light;

void main() {
    // ambient
    vec3 ambient = light.ambient * texture(material.diffuse, tex_coords).rgb;

    // diffuse
    vec3 norm = normalize(normal);
    vec3 light_dir = normalize(light.position - frag_pos);
    float diff = max(dot(norm, light_dir), 0.0f);
    vec3 diffuse = light.diffuse * diff * texture(material.diffuse, tex_coords).rgb;
    // specular
    vec3 view_dir = normalize(view_pos - frag_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0f), material.shininess);
    vec3 specular = light.specular * spec * texture(material.specular, tex_coords).rgb;

    float theta = dot(light_dir, normalize(-light.direction));
    float epsilon = light.cut_off - light.out_cut_off;
    float intensity = clamp((theta - light.out_cut_off) / epsilon, 0.0f, 1.0f);

    diffuse *= intensity;
    specular *= intensity;

    // attenuation
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0f / (light.constant + light.linear * distance + light.quadratic * distance * distance);

    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;

    vec3 result = ambient + diffuse + specular;
    color = vec4(result, 1.0f);
}