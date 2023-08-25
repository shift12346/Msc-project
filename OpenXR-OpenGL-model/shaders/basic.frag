 #version 410

    in vec3 PSVertexColor;
    out vec4 FragColor;

    void main() {
       
       vec3 whiteColor = vec3(1,1,1);
       FragColor = vec4(whiteColor, 1);
    }