 #version 410
 #extension GL_ARB_geometry_shader4 : enable

 void main()
 {
    // increment variable
    int i;
    vec4 vertex;
   
    for(i = 0; i < gl_VerticesIn; i++)
    {
      gl_Position = gl_PositionIn[i];
      EmitVertex();
    }

    EndPrimitive();

  }