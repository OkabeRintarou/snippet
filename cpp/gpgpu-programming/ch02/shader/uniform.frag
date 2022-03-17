uniform float v_time;

void main(void) {
    float r = 0.9 * sin(0.0 + v_time * 0.05) + 1.0;
    float g = 0.9 * cos(0.33 + v_time * 0.05) + 1.0;
    float b = 0.9 * sin(0.67 + v_time * 0.05) + 1.0;
    gl_FragColor = vec4(r / 2, g / 2, b / 2, 1.0);
}