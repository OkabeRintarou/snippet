#include <cstdio>
#include "util.h"
#include "opengl.h"

void runMainLoop(int val);

int main(int argc,char *argv[])
{
	glutInit(&argc,argv);
	glutInitContextVersion(2,1);

	glutInitDisplayMode(GLUT_DOUBLE);
	glutInitWindowSize(SCREEN_WIDTH,SCREEN_HEIGHT);
	glutCreateWindow("OpenGL");

	if(!initGL()){
		fprintf(stderr,"Unable to initialize graphics library!\n");
		return 1;
	}	

	// Set rendering function
	glutDisplayFunc(render);
		
	// Set main loop
	glutTimerFunc(1000.0 / SCREEN_FPS,runMainLoop,0);

	// Start GLUT main loop
	glutMainLoop();

	return 0;
}

void runMainLoop(int val)
{
	update();
	render();

	// Run frame one more time
	glutTimerFunc(1000.0 / SCREEN_FPS,runMainLoop,val);
}
