#include <stdlib.h>

#include <SDL2/SDL.h>

int main(int argc,char* argv[])
{
	if(SDL_Init(SDL_INIT_EVERYTHING) < 0) {
		fprintf(stderr,"SDL could not initialize! SDL_Error:%s\n",SDL_GetError());
		return -1;
	}
	atexit(SDL_Quit);
	SDL_Window* window = SDL_CreateWindow("ch01",SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,640,480,SDL_WINDOW_SHOWN);
	SDL_Delay(2000);
	SDL_DestroyWindow(window);
	return 0;
}
