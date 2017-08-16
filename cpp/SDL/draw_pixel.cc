#include <SDL2/SDL.h>
#include <cstdio>
#include <cmath>

SDL_Window* window = NULL;
SDL_Renderer* render = NULL;

int Init()
{
	SDL_Init(SDL_INIT_VIDEO);
	window = SDL_CreateWindow("pixel",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,400,240,SDL_WINDOW_SHOWN);
	if(window == NULL) {
		return -1;
	}

	render = SDL_CreateRenderer(window,-1,SDL_RENDERER_ACCELERATED);
	if(render == NULL) {
		return -1;
	}
	SDL_SetRenderDrawColor(render,0xff,0xff,0xff,0xff);
	return 0;
}

void Destroy()
{
	SDL_DestroyRenderer(render);
	SDL_DestroyWindow(window);
	SDL_Quit();
}

int main()
{
	if(Init() < 0) {
		fprintf(stderr,"Init error:%s\n",SDL_GetError());
		return -1;
	}

	SDL_RenderClear(render);

	for(int h = 0;h < 240;h++) {
		for(int w = 0;w < 400;w++) {
			SDL_SetRenderDrawColor(render,rand() % 256,rand() % 256,rand() % 256,0xff);
			SDL_RenderDrawPoint(render,w,h);
		}
	}
	SDL_RenderPresent(render);
	SDL_Delay(10000);
	Destroy();
	return 0;
}
