#include <stdio.h>
#include <stdlib.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

static const int Width = 640;
static const int Height = 480;

int main()
{
	if(SDL_Init(SDL_INIT_EVERYTHING) < 0) {
		fprintf(stderr,"SDL init faild!SDL_Error:%s\n",SDL_GetError());
		return -1;
	}
	atexit(SDL_Quit);

	SDL_Window* window = SDL_CreateWindow("ch07",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,
	Width,Height,SDL_WINDOW_SHOWN);
	if(window == NULL) {
		fprintf(stderr,"Window could not be created! SDL_Error:%s\n",SDL_GetError());
		return -1;
	}

	SDL_Renderer* render = SDL_CreateRenderer(window,-1,SDL_RENDERER_ACCELERATED);
	if(render == NULL) {
		fprintf(stderr,"Renderer could not be created! SDL_Error:%s\n",SDL_GetError());
		return -1;
	}

	SDL_Texture* texture = NULL;
	SDL_Surface* surface = NULL;

	surface = IMG_Load("texture.png");
	if(!surface) {
		fprintf(stderr,"Unable to load image! SDL_Error:%s\n",SDL_GetError());
		return -1;
	}
	texture = SDL_CreateTextureFromSurface(render,surface);
	if(texture == NULL) {
		fprintf(stderr,"Unable to create texture from surface! SDL_Error:%s\n",SDL_GetError());
		return -1;
	}
	SDL_FreeSurface(surface);
	surface = NULL;
	
	bool quit = false;
	SDL_Event event;
	while(!quit) {
		while(SDL_PollEvent(&event) != 0) {
			if(event.type == SDL_QUIT) quit = true;
		}
		SDL_RenderClear(render);
		SDL_RenderCopy(render,texture,NULL,NULL);
		SDL_RenderPresent(render);
	}
	SDL_DestroyRenderer(render);
	SDL_DestroyWindow(window);
	
	return 0;
}
