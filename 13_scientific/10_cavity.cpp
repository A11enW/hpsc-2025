#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>


using namespace std;
typedef vector<vector<float>> matrix;

int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  matrix u(ny,vector<float>(nx));
  matrix v(ny,vector<float>(nx));
  matrix p(ny,vector<float>(nx));
  matrix b(ny,vector<float>(nx));
  matrix un(ny,vector<float>(nx));
  matrix vn(ny,vector<float>(nx));
  matrix pn(ny,vector<float>(nx));
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j][i] = 0;
      v[j][i] = 0;
      p[j][i] = 0;
      b[j][i] = 0;
    }
  }
  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n=0; n<nt; n++) {
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        //Compute the sub-item that needed for calculating b[j][i]
	const double dudx = (u[j][i + 1] - u[j][i - 1]) / (2.0 * dx);
    	const double dvdy = (v[j + 1][i] - v[j - 1][i]) / (2.0 * dy);
    	const double dudy = (u[j + 1][i] - u[j - 1][i]) / (2.0 * dy);
    	const double dvdx = (v[j][i + 1] - v[j][i - 1]) / (2.0 * dx);
	const double term1 = (1.0 / dt) * (dudx + dvdy);
    	const double term2 = dudx * dudx;
    	const double term3 = 2.0 * dudy * dvdx;
    	const double term4 = dvdy * dvdy;
	//1 Compute b[j][i]
	b[j][i] = static_cast<float>(rho * (term1 - term2 - term3 - term4));
      }
    }
    for (int it=0; it<nit; it++) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
	  pn[j][i] = p[j][i];
      for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
	  //2 Compute p[j][i]
	  p[j][i] = static_cast<float>(
            (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
             dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
             b[j][i] * dx * dx * dy * dy) /
            (2.0 * (dx * dx + dy * dy)));
	  }
      }
      for (int j=0; j<ny; j++) {
        //3 Compute p[j][0] and p[j][nx-1]
	p[j][0]      = p[j][1];        // left part：∂p/∂x = 0
        p[j][nx-1]   = p[j][nx-2];     // right part：∂p/∂x = 0
      }
      for (int i=0; i<nx; i++) {
	//4 Compute p[0][i] and p[ny-1][i]
	p[0][i]      = p[1][i];   // Bottom (y = 0)：∂p/∂y = 0
    	p[ny-1][i]   = 0.0f;      // Top (y = 2)：for p = 0
      }
    }
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j][i] = u[j][i];
	vn[j][i] = v[j][i];
      }
    }
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
	//5 Compute u[j][i] and v[j][i]
	const double unji = un[j][i];
    	const double vnji = vn[j][i];
	//for u
	const double conv_u_x = unji * dt / dx * (unji - un[j][i - 1]);
    	const double conv_u_y = vnji * dt / dy * (unji - un[j - 1][i]);
    	const double pres_u   = dt / (2.0 * rho * dx) * (p[j][i + 1] - p[j][i - 1]);
    	const double diff_u_x = nu * dt / (dx * dx) * (un[j][i + 1] - 2.0 * unji + un[j][i - 1]);
    	const double diff_u_y = nu * dt / (dy * dy) * (un[j + 1][i] - 2.0 * unji + un[j - 1][i]);
	u[j][i] = static_cast<float>(unji- conv_u_x - conv_u_y - pres_u + diff_u_x + diff_u_y);
	//for v
	const double conv_v_x = unji * dt / dx * (vnji       - vn[j][i - 1]);
    	const double conv_v_y = vnji * dt / dy * (vnji       - vn[j - 1][i]);
    	const double pres_v   = dt / (2.0 * rho * dx) * (p[j + 1][i] - p[j - 1][i]);
    	const double diff_v_x = nu * dt / (dx * dx) * (vn[j][i + 1] - 2.0 * vnji + vn[j][i - 1]);
    	const double diff_v_y = nu * dt / (dy * dy) * (vn[j + 1][i] - 2.0 * vnji + vn[j - 1][i]);
	v[j][i] = static_cast<float>(vnji - conv_v_x - conv_v_y - pres_v + diff_v_x + diff_v_y);
      }
    }
    for (int j=0; j<ny; j++) {
      //6 Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
      // Left
      u[j][0]      = 0.0f;      
      v[j][0]      = 0.0f;   
      // Right
      u[j][nx-1]   = 0.0f;     
      v[j][nx-1]   = 0.0f;      
    }
    for (int i=0; i<nx; i++) {
      //7 Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
      // Bottom Condition
      u[0][i]    = 0.0f;
      v[0][i]    = 0.0f;
      // Top Condition
      u[ny-1][i] = 1.0f;
      v[ny-1][i] = 0.0f;
    }
    if (n % 10 == 0) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j][i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j][i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j][i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
}
