#include "vec3.h"
#include <iostream>

using namespace std;

int main() {
    const int nx = 200;
    const int ny = 100;
    cout << "P3\n"
         << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            vec3 col(float(i) / float(nx), float(j) / float(ny), 0.2f);
            int ir = int(255.99f * col.r());
            int ig = int(255.99f * col.g());
            int ib = int(255.99f * col.b());
            cout << ir << " " << ig << " " << ib << '\n';
        }
    }
    return 0;
}
