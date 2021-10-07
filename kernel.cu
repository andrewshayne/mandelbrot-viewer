
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SFML/Graphics.hpp>

#include <stdio.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

//__device__ __constant__ int THREAD_COUNT = 100;


int mandelbrot_type = 1;
int MAX_ITERATIONS = 1000;
int LINES = 500;
const int WIDTH = 1024;
const int HEIGHT = 1024;

__device__ __constant__ int const_WIDTH = WIDTH;
__device__ __constant__ int const_HEIGHT = HEIGHT;
//sf::Uint8* buff;
uint8_t* buff;

//iteration buffer
int* iteration_buff;


long double x_lower = -1.5;
long double x_upper = 1.5;
long double y_lower = -1.5;
long double y_upper = 1.5;

//set point to zoom in on and window size around it

// origin
long double shrink_factor = 1.5;
long double x_origin = 0.0;
long double y_origin = 0.0;

// full view of Z^2
//long double shrink_factor = 1.5;
//long double x_origin = -0.75;
//long double y_origin = 0.0;

// cool spiral coords!
//long double shrink_factor = 0.000046;
//long double x_origin = -0.530859;
//long double y_origin = -0.592524;



//consts for row coloring function below...
__device__ __constant__ int const_rgb_1[3];
__device__ __constant__ int const_rgb_2[3];
__device__ __constant__ int const_rgb_diff[3];

int rgb_1[3] = { 124, 0, 181 };
int rgb_2[3] = { 230, 192, 25 };
int rgb_diff[3] = { rgb_2[0] - rgb_1[0], rgb_2[1] - rgb_1[1], rgb_2[2] - rgb_1[2] };


//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t setBuffWithCuda(uint8_t* buff, int* iteration_buff, const long double x_lower, const long double x_upper, const long double y_lower, const long double y_upper, const int MAX_ITERATIONS, const int mandelbrot_type);

__global__ void setPixelKernel(uint8_t* buff, int* iteration_buff, const long double x_lower, const long double x_upper, const long double y_lower, const long double y_upper, const int MAX_ITERATIONS, const int mandelbrot_type)
{
	//derive i and j from index...
	int i = blockIdx.x * blockDim.x + threadIdx.x;// index;
	int j = blockIdx.y * blockDim.y + threadIdx.y;// index;


	long double x = x_lower + i / (const_WIDTH - 1.0)*(x_upper - x_lower);
	long double y = y_lower + j / (const_HEIGHT - 1.0)*(y_upper - y_lower);
	//complex Z(0.0, 0.0);
	//complex C(x, y);

	long double Z_a = 0.0;
	long double Z_b = 0.0;
	long double C_a = x;
	long double C_b = y;

	int count;
	long double a;
	long double b;
	for (count = 0; count < MAX_ITERATIONS && (Z_a*Z_a + Z_b*Z_b) < 4.0; count++) {
		a = Z_a;
		b = Z_b;

		long double a2 = a * a;
		long double b2 = b * b;
		long double a4 = a2 * a2;
		long double b4 = b2 * b2;

		switch (mandelbrot_type)
		{
		case 0:
			// burning ship
			Z_a = a2 - b2 + C_a;
			Z_b = 2.0 * a * b;
			if (Z_b < 0) Z_b = -Z_b;
			Z_b += C_b;
			break;
		case 1:
			// Z = Z^2 + C;
			Z_a = (a*a - b * b) + C_a;
			Z_b = (2.0 * a * b) + C_b;
			break;

			// Mandelbar (Z = Z_bar^2 + C)
			//Z_a = (a*a - b*b) + C_a;
			//Z_b = (-2.0 * a * b) + C_b;

		case 2:
			// Z = Z^3 + C;
			Z_a = a*(a2 - 3.0*b2) + C_a;
			Z_b = b*(3.0*a2 - b2) + C_b;
			break;

		case 3:
			// Z = Z^4 + C;
			Z_a = a2*a2 -6.0*a2*b2 + b2*b2 + C_a;
			Z_b = 4.0*a*b*(a2 - b2) + C_b;
			break;

		case 5:
			// Z = Z^6 + C;
			Z_a = a4*a2 - 15.0*a4*b2 + 15.0*a2*b4 - b4*b2 + C_a;
			Z_b = 6.0*a4*a*b -20.0*a2*a*b2*b + 6*a*b4*b + C_b;
		}

	}

	iteration_buff[j*const_HEIGHT + i] = count;

	//sf::Color myColor = sf::Color(
	//	(const_rgb_diff[0] * count / M) + const_rgb_1[0],
	//	(const_rgb_diff[1] * count / M) + const_rgb_1[1],
	//	(const_rgb_diff[2] * count / M) + const_rgb_1[2]
	//);


	//if (count != MAX_ITERATIONS) {
	//	buff[4 * (j*const_HEIGHT + i)]	   = (const_rgb_diff[0] * count / MAX_ITERATIONS) + const_rgb_1[0];
	//	buff[4 * (j*const_HEIGHT + i) + 1] = (const_rgb_diff[1] * count / MAX_ITERATIONS) + const_rgb_1[1];
	//	buff[4 * (j*const_HEIGHT + i) + 2] = (const_rgb_diff[2] * count / MAX_ITERATIONS) + const_rgb_1[2];
	//	buff[4 * (j*const_HEIGHT + i) + 3] = 255;
	//}

	if (count != MAX_ITERATIONS) {
		float H = (count+360) % 360;
		float S = 0.5;
		float V = 0.75;
		float C = V * S;

		float partAbs = fmod((double)(H / 60.0), 2.0) - 1.0;
		partAbs = (partAbs > 0 ? partAbs : -partAbs);
		float X = C * (1 - partAbs);
		float m = V - C;

		float Rp = 0;
		float Gp = 0;
		float Bp = 0;
		//int R; int G; int B;

		if (H < 60) {
			Rp = C; Gp = X; Bp = 0;
		} else if (H < 120) {
			Rp = X; Gp = C; Bp = 0;
		} else if (H < 180) {
			Rp = 0; Gp = C; Bp = X;
		} else if (H < 240) {
			Rp = 0; Gp = X; Bp = C;
		} else if (H < 300) {
			Rp = X; Gp = 0; Bp = C;
		} else { // H < 360
			Rp = C; Gp = 0; Bp = X;
		}

		buff[4 * (j*const_HEIGHT + i)]	   = (Rp+m)*255;
		buff[4 * (j*const_HEIGHT + i) + 1] = (Gp+m)*255;
		buff[4 * (j*const_HEIGHT + i) + 2] = (Bp+m)*255;
		buff[4 * (j*const_HEIGHT + i) + 3] = 255;
	}
	else {
		buff[4 * (j*const_HEIGHT + i)]     = 0;
		buff[4 * (j*const_HEIGHT + i) + 1] = 0;
		buff[4 * (j*const_HEIGHT + i) + 2] = 0;
		buff[4 * (j*const_HEIGHT + i) + 3] = 0;
	}
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void cpuColorPixel(int count, int i, int j, int color_tastes_like_honey)
{
	if (count != MAX_ITERATIONS) {
		count += color_tastes_like_honey;
		float H = (count+360) % 360;
		float S = 0.5;
		float V = 0.75;
		float C = V * S;

		float partAbs = fmod((double)(H / 60.0), 2.0) - 1.0;
		partAbs = (partAbs > 0 ? partAbs : -partAbs);
		float X = C * (1 - partAbs);
		float m = V - C;

		float Rp = 0;
		float Gp = 0;
		float Bp = 0;
		//int R; int G; int B;

		if (H < 60) {
			Rp = C; Gp = X; Bp = 0;
		} else if (H < 120) {
			Rp = X; Gp = C; Bp = 0;
		} else if (H < 180) {
			Rp = 0; Gp = C; Bp = X;
		} else if (H < 240) {
			Rp = 0; Gp = X; Bp = C;
		} else if (H < 300) {
			Rp = X; Gp = 0; Bp = C;
		} else { // H < 360
			Rp = C; Gp = 0; Bp = X;
		}


		buff[4 * (j*HEIGHT + i)]	   = (Rp+m)*255;
		buff[4 * (j*HEIGHT + i) + 1] = (Gp+m)*255;
		buff[4 * (j*HEIGHT + i) + 2] = (Bp+m)*255;
		buff[4 * (j*HEIGHT + i) + 3] = 255;
	}
	else {
		buff[4 * (j*HEIGHT + i)]     = 0;
		buff[4 * (j*HEIGHT + i) + 1] = 0;
		buff[4 * (j*HEIGHT + i) + 2] = 0;
		buff[4 * (j*HEIGHT + i) + 3] = 0;
	}
}

// Helper function for using CUDA to add vectors in parallel. - MODIFY FOR MANDELBROT
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
cudaError_t setBuffWithCuda(uint8_t* buff, int* iteration_buff,
	const long double x_lower, const long double x_upper, const long double y_lower, const long double y_upper,
	const int MAX_ITERATIONS, const int mandelbrot_type)
{
	uint8_t* dev_buff;
	int* dev_iteration_buff;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	// * Allocate for mandelbrot screen buffer
	unsigned int size = WIDTH * HEIGHT;
    cudaStatus = cudaMalloc((void**)&dev_buff, size * 4 * sizeof(uint8_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// * Allocate for mandelbrot screen iteration buffer
	cudaStatus = cudaMalloc((void**)&dev_iteration_buff, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


	// NOT NECESSARY- every computation is completely fresh
	/*
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	*/

    // Launch a kernel on the GPU with one thread for each element.
	dim3 blocksPerGrid(32, 32, 1);
	dim3 threadsPerBlock(32, 32, 1);
	setPixelKernel <<<blocksPerGrid, threadsPerBlock>>> (dev_buff, dev_iteration_buff, x_lower, x_upper, y_lower, y_upper, MAX_ITERATIONS, mandelbrot_type);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(buff, dev_buff, size * 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	// Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(iteration_buff, dev_iteration_buff, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_buff);
    cudaFree(dev_iteration_buff);
    
    return cudaStatus;
}




int main()
{
	int color_mod = 0;
	int color_mod_incr = 0;

	bool hideText = false;
	bool drawLines = false;
	bool refreshIterations = false;

	cudaMemcpyToSymbol(const_rgb_1, rgb_1, sizeof(rgb_1));
	cudaMemcpyToSymbol(const_rgb_2, rgb_2, sizeof(rgb_2));
	cudaMemcpyToSymbol(const_rgb_diff, rgb_diff, sizeof(rgb_diff));

	sf::Clock clock;
	std::setprecision(10);

	//x_origin = -0.59990625;
	//y_origin = -0.4290703125;
	//shrink_factor = 0.001;

	x_lower = x_origin - shrink_factor;
	x_upper = x_origin + shrink_factor;
	y_lower = y_origin - shrink_factor;
	y_upper = y_origin + shrink_factor;

	long double x_interval = (x_upper - x_lower) / (long double)WIDTH;
	long double y_interval = (y_upper - y_lower) / (long double)HEIGHT;
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Mandelbrot Viewer");

	//buff = new sf::Uint8[WIDTH * HEIGHT * 4];
	buff = new uint8_t[WIDTH * HEIGHT * 4];
	iteration_buff = new int[WIDTH * HEIGHT];
	sf::Image buf;
	//buf.create(WIDTH, HEIGHT, sf::Color::Black);

	sf::Sprite mSprite;
	sf::Texture mTexture;

	sf::Font font;
	font.loadFromFile("Resources/BELLB.TTF");

	//TEXT
	sf::Text centerText;
	centerText.setFont(font);
	centerText.setPosition({10.f,0.f});

	sf::Text rectText;
	rectText.setFont(font);
	rectText.setPosition({10.f,30.f});

	sf::Text colorRotateText;
	colorRotateText.setFont(font);
	colorRotateText.setPosition({10.f,90.f});

	sf::Text iterationText;
	iterationText.setFont(font);
	iterationText.setPosition({10.f,150.f});

	//controls...
	sf::Text lineText;
	lineText.setFont(font);
	lineText.setPosition({10.f,210.f});

	//hide text
	sf::Text hideTextText;
	hideTextText.setFont(font);
	hideTextText.setPosition({740.f,0.f});


	centerText.setString("Center: " + to_string(x_origin) + ", " + to_string(y_origin));
	rectText.setString("ViewPort Size: " + to_string(shrink_factor));
	colorRotateText.setString("Color Rotation Speed (\'A\'/\'B\'): 0");
	iterationText.setString("Iterations (\'Up\'/\'Down\'): " + to_string(MAX_ITERATIONS));
	lineText.setString("Toggle Line Mode: \'L\'");
	hideTextText.setString("Press H to hide text");

	setBuffWithCuda(buff, iteration_buff, x_lower, x_upper, y_lower, y_upper, MAX_ITERATIONS, mandelbrot_type);
	buf.create(WIDTH, HEIGHT, (const uint8_t*)buff);
	mTexture.loadFromImage(buf);
	mSprite.setTexture(mTexture);


	//if I click on the screen, make it "shrink" the rectangle window by factor of 10 around that point!

	bool currentlyRendering = false;

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();

			bool changedMandelbrotType = false;

			//key pressed
			if (event.type == sf::Event::KeyPressed && !currentlyRendering)
			{
				//toggle text display
				if (event.key.code == sf::Keyboard::H)
				{
					hideText = !hideText;
				}

				//alter color rotation speed
				if (event.key.code == sf::Keyboard::A)
				{
					color_mod_incr++;
					printf("increase color mod: %i\n", color_mod_incr);
					colorRotateText.setString("Color Rotation Speed: " + to_string(color_mod_incr));
				}
				if (event.key.code == sf::Keyboard::B)
				{
					color_mod_incr--;
					//if (color_mod_incr < 0) color_mod_incr = 0;
					printf("decrease color mod: %i\n", color_mod_incr);
					colorRotateText.setString("Color Rotation Speed: " + to_string(color_mod_incr));
				}

				//double/half iteration count
				//ooooooor +/- 1000
				if (event.key.code == sf::Keyboard::Up)
				{
					MAX_ITERATIONS += 1000;
					iterationText.setString("Iterations: " + to_string(MAX_ITERATIONS));
					refreshIterations = true;
				}
				if (event.key.code == sf::Keyboard::Down)
				{
					MAX_ITERATIONS -= 1000;
					if (MAX_ITERATIONS <= 0)
						MAX_ITERATIONS = 2;
					iterationText.setString("Iterations: " + to_string(MAX_ITERATIONS));
					refreshIterations = true;
				}

				//toggle line render
				if (event.key.code == sf::Keyboard::L)
				{
					drawLines = !drawLines;
					//lineText.setString("Toggle Lines:")
				}

				if (event.key.code == sf::Keyboard::Num0)
				{
					mandelbrot_type = 0;
					changedMandelbrotType = true;
				}
				if (event.key.code == sf::Keyboard::Num1)
				{
					mandelbrot_type = 1;
					changedMandelbrotType = true;
				}
				if (event.key.code == sf::Keyboard::Num2)
				{
					mandelbrot_type = 2;
					changedMandelbrotType = true;
				}
				if (event.key.code == sf::Keyboard::Num3)
				{
					mandelbrot_type = 3;
					changedMandelbrotType = true;
				}
				if (event.key.code == sf::Keyboard::Num5)
				{
					mandelbrot_type = 5;
					changedMandelbrotType = true;
				}

			}

			//mouse pressed
			if (((event.type == sf::Event::MouseButtonPressed && (event.mouseButton.button == sf::Mouse::Left || event.mouseButton.button == sf::Mouse::Right)) || event.type == sf::Event::MouseWheelScrolled || refreshIterations || changedMandelbrotType || changedMandelbrotType) && !currentlyRendering) {
				clock.restart();

				//set origin and reduce shrink!

				//scroll
				if (event.type == sf::Event::MouseWheelScrolled)
				{
					if (event.mouseWheelScroll.delta > 0)
						shrink_factor *= 0.5;
					else if (event.mouseWheelScroll.delta < 0)
						shrink_factor *= 2.0;
				}

				//click
				if (event.type == sf::Event::MouseButtonPressed)
				{
					if (event.mouseButton.button == sf::Mouse::Left)
						shrink_factor *= 0.66666666667;
					else if (event.mouseButton.button == sf::Mouse::Right)
						shrink_factor *= 1.5;
				}

				//do not recalculate transformation if we are only refreshing
				if (!refreshIterations)
				{
					//get mouse pixel coords
					int x_window = sf::Mouse::getPosition(window).x;
					int y_window = sf::Mouse::getPosition(window).y;
	
					x_origin = (x_upper - x_lower)*x_window / WIDTH + x_lower;
					y_origin = (y_upper - y_lower)*y_window / HEIGHT + y_lower;

					x_lower = x_origin - shrink_factor;
					x_upper = x_origin + shrink_factor;
					y_lower = y_origin - shrink_factor;
					y_upper = y_origin + shrink_factor;

					//set origin to 0 if changed type
					if (changedMandelbrotType)
					{
						x_origin = 0;
						y_origin = 0;
						shrink_factor = 1.5f;

						x_lower = x_origin - shrink_factor;
						x_upper = x_origin + shrink_factor;
						y_lower = y_origin - shrink_factor;
						y_upper = y_origin + shrink_factor;
					}

					centerText.setString("Center: " + to_string(x_origin) + ", " + to_string(y_origin));
					rectText.setString("ViewPort Size: " + to_string(shrink_factor));
				}

				currentlyRendering = true;

				setBuffWithCuda(buff, iteration_buff, x_lower, x_upper, y_lower, y_upper, MAX_ITERATIONS, mandelbrot_type);

				// uint8_t alias to sf::Uint8
				buf.create(WIDTH, HEIGHT, (const sf::Uint8*)buff);
				//buf.create(WIDTH, HEIGHT, (const uint8_t*)buff);
				mTexture.loadFromImage(buf);
				mSprite.setTexture(mTexture); //code gets OOF'd right here when iterations get too high
				currentlyRendering = false;

				sf::Time renderTime = clock.getElapsedTime();
				printf("time to render: %f\n", renderTime.asSeconds());

				refreshIterations = false;
			}

		}

		color_mod += color_mod_incr;
		if (color_mod > 360)
		{
			color_mod = 0;
		}
		else if (color_mod < 0)
		{
			color_mod = 360;
		}

		//color rotate each pixel
		if (color_mod_incr != 0)
		{
			for (int i = 0; i < HEIGHT; i++)
			{
				for (int j = 0; j < WIDTH; j++)
				{
					cpuColorPixel(iteration_buff[j*HEIGHT + i], i, j, color_mod);
				}
			}
		}

		//set trippy stuff for rendering
		buf.create(WIDTH, HEIGHT, (const sf::Uint8*)buff);
		mTexture.loadFromImage(buf);
		mSprite.setTexture(mTexture);

		window.clear();
		window.draw(mSprite);

		//LINES
		if (drawLines)
		{

			//get mouse pixel coords
			float x_window = sf::Mouse::getPosition(window).x;
			float y_window = sf::Mouse::getPosition(window).y;

			float x = (x_upper - x_lower)*x_window / WIDTH + x_lower;
			float y = (y_upper - y_lower)*y_window / HEIGHT + y_lower;

			float Z_a = 0.0;
			float Z_b = 0.0;
			float C_a = x;
			float C_b = y;

			float a;
			float b;

			sf::Vertex* line = new sf::Vertex[LINES*2];

			for (int i = 0; i < LINES; i++)
			{
				a = Z_a;
				b = Z_b;

				long double a2 = a * a;
				long double b2 = b * b;
				long double a4 = a2 * a2;
				long double b4 = b2 * b2;

				switch (mandelbrot_type)
				{
				case 0:
					// burning ship
					Z_a = a2 - b2 + C_a;
					Z_b = 2.0 * a * b;
					if (Z_b < 0) Z_b = -Z_b;
					Z_b += C_b;
					break;
				case 1:
					// Z = Z^2 + C;
					Z_a = (a*a - b * b) + C_a;
					Z_b = (2.0 * a * b) + C_b;
					break;

					// Mandelbar (Z = Z_bar^2 + C)
					//Z_a = (a*a - b*b) + C_a;
					//Z_b = (-2.0 * a * b) + C_b;

				case 2:
					// Z = Z^3 + C;
					Z_a = a * (a2 - 3.0*b2) + C_a;
					Z_b = b * (3.0*a2 - b2) + C_b;
					break;

				case 3:
					// Z = Z^4 + C;
					Z_a = a2 * a2 - 6.0*a2*b2 + b2 * b2 + C_a;
					Z_b = 4.0*a*b*(a2 - b2) + C_b;
					break;

				case 5:
					// Z = Z^6 + C;
					Z_a = a4 * a2 - 15.0*a4*b2 + 15.0*a2*b4 - b4 * b2 + C_a;
					Z_b = 6.0*a4*a*b - 20.0*a2*a*b2*b + 6 * a*b4*b + C_b;
				}

				line[i * 2] = sf::Vertex({ x_window, y_window });

				x_window = (a - x_lower)*WIDTH / (x_upper - x_lower);
				y_window = (b - y_lower)*HEIGHT / (y_upper - y_lower);

				line[i * 2 + 1] = sf::Vertex({ x_window, y_window });
			}
			window.draw(line, LINES, sf::Lines);
		}

		//TEXT
		if (!hideText)
		{
			window.draw(centerText);
			window.draw(rectText);
			window.draw(colorRotateText);
			window.draw(iterationText);
			window.draw(lineText);
			window.draw(hideTextText);
		}

		//DISPLAY
		window.display();
	}

	return 0;
}


