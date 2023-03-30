//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1);		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

float hdot(vec3 p, vec3 q) {
	return p.x * q.x + p.y * q.y - p.z * q.z;
}

bool onMinkowsky(vec3 p) {
	float threshold = 1e-5;
	return fabs(hdot(p, p) + 1.0f) < threshold;
}

bool vectorLegalAtPoint(vec3 p, vec3 v) {
	float threshold = 1e-5;
	return fabs(hdot(p, v)) < threshold;
}

void calculateW(vec3& p) {
	p.z = sqrtf(p.x * p.x + p.y * p.y + 1);
}

void calculateW(vec3 p, vec3& v) {
	if (hdot(p, v) == 0) return;
	v.z = (p.x * v.x + p.y * v.y) / p.z;
}

float hlength(vec3 v) {
	return sqrtf(hdot(v, v));
}

vec3 hnormalize(vec3 v) {
	return v * 1 / length(v);
}

vec3 movePoint(vec3 p, vec3 v0, float t) {
	return p * coshf(t) + v0 * sinhf(t);
}

vec3 moveVector(vec3 p, vec3 v0, float t) {
	return p * sinhf(t) + v0 * coshf(t);
}

vec3 perpendicularVector(vec3 p, vec3 v) {
	v = hnormalize(v);
	p.z = -p.z;
	//p = hnormalize(p);
	v.z = -v.z;
	float nx = p.y * v.z - p.z * v.y;
	float ny = p.z * v.x - p.x * v.z;
	float nz = p.x * v.y - p.y * v.x;
	vec3 val = vec3(nx, ny, nz);
	val = hnormalize(val);
	calculateW(p, val);
	return val;
}

vec3 rotate(vec3 p, vec3 v, float fi) {
	v = hnormalize(v);
	vec3 vp = perpendicularVector(p, v);
	vec3 rot = v * cosf(fi) + vp * sinf(fi);
	rot = hnormalize(rot);
	calculateW(p, rot);
	return rot;
}


float hdistance(vec3 p, vec3 q) {
	return acosh(-hdot(p, q));
}

vec3 project(vec3 p) {
	return vec3(p.x / (p.z + 1), p.y / ( p.z + 1), 0);
}




GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

const int NV = 100;





class Circle {
private:
	vec3 center;
	unsigned int vao;
	vec2 vertices[NV];
public:
	void Create() {
		for (int i = 0; i < NV; i++) {
			float fi = i * 2 * M_PI / NV;
			vec2 p = vec2(cosf(fi), sinf(fi));
			vertices[i] = p;
		}

		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec2) * NV,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}

	void Draw() {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 0.0f, 0.0f, 0.0f);
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, NV /*# Elements*/);
	}
};

class HCircle {
private:
	vec3 center;
	unsigned int vao;
	vec3 vertices[NV];
public:
	void Create(vec3 ctr) {
		glGenVertexArrays(1, &vao);	// get 1 vao id
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		center = ctr;
		calculateW(center);
		vec3 V = { 0.0f ,0.0f,0.0f };
		for (int i = 0; i < NV; i++) {
			float fi = i * 2 * M_PI / NV;
			V = vec3(1.0f, 0.0f, 0.0f);
			calculateW(center, V);
			V = rotate(center, V, fi);
			vec3 newpoint = movePoint(center, V, 0.5);
			calculateW(newpoint);
			if (!onMinkowsky(newpoint)) {
				printf("SOS");
			}
			vertices[i] = project(newpoint);
			
		}
		glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
			sizeof(vec3) * NV,  // # bytes
			vertices,	      	// address
			GL_STATIC_DRAW);	// we do not change later

		glEnableVertexAttribArray(0);  // AttribArray 0
		glVertexAttribPointer(0,       // vbo -> AttribArray 0
			3, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
			0, NULL); 		     // stride, offset: tightly packed
	}

	void Draw(std::vector<float> colors) {
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, colors[0], colors[1],colors[2]);
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, NV /*# Elements*/);
	}
};

class Hami {
private:
	HCircle body;
	/*HCircle leftEye;
	HCircle rightEye;
	HCircle mouth;*/
	vec3 direction;
	vec3 location;
public:
	void Create(vec3 ctr) {
		location = ctr;
		calculateW(location);
		body.Create(location);
		direction = vec3(1.0f, 0.0f, 0.0f);
		calculateW(location, direction);
		//direction = hnormalize(direction);
	}

	void Move() {
		vec3 newLocation = movePoint(location, direction, 0.01f);
		calculateW(newLocation);
		vec3 newDirection = moveVector(newLocation, direction, 0.01f);
		calculateW(newLocation, newDirection);
		newDirection = rotate(newLocation, newDirection, M_PI/100.0f);
		location = newLocation;
		calculateW(location);
		if (!onMinkowsky(location))
			printf("BAJ\n");
		direction = newDirection;
		calculateW(location, direction);

		body.Create(location);
		//body.Draw(std::vector<float>{ 0.0f, 1.0f, 0.0f });
		glutPostRedisplay();
	}

	void Draw(std::vector<float> colors) {
		body.Draw(colors);
	}
};

Circle circle;
HCircle redCircle;
HCircle greenCircle;
Hami greenHami;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	circle.Create();
	redCircle.Create(vec3(0.0f, 0.0f, 0.0f));
	//greenCircle.Create(vec3(2.0f, 0.0f, 0.0f));
	greenHami.Create(vec3(1.0f, 0.0f, 0.0f));

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.125, 0.125, 0.125, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	circle.Draw();
	redCircle.Draw(std::vector<float>{ 1.0f, 0.0f, 0.0f});
	//greenCircle.Draw(std::vector<float>{ 0.0f, 1.0f, 0.0f });
	greenHami.Draw(std::vector<float>{ 0.0f, 1.0f, 0.0f });
	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	/*char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;


	}*/
	//greenHami.Move();
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	/*if(time % 2 == 0)
		greenHami.Move();
}*/
