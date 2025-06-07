//parallelize the 10_cavity.cpp with MPI

#include <mpi.h>
#include <vector>
#include <fstream>
#include <cassert>

using std::vector;
using std::ofstream;

//define the constrant value
constexpr int nx  = 41;
constexpr int ny  = 41;
constexpr int nt  = 500;
constexpr int nit = 50;

constexpr double dx  = 2.0 / (nx - 1);
constexpr double dy  = 2.0 / (ny - 1);
constexpr double dt  = 0.01;
constexpr double rho = 1.0;
constexpr double nu  = 0.02;


inline int IDX(int j, int i, int lnx) { return j * lnx + i; }

void exchange_ghost(vector<float>& buf, int lny, int lnx,
                    int north, int south, MPI_Comm comm)
{
    const int row_sz = lnx;
    MPI_Request req[4];

    // north = rank-1, south = rank+1
    MPI_Irecv(&buf[IDX(0, 0, lnx)], row_sz, MPI_FLOAT, north, 0, comm, &req[0]);
    MPI_Irecv(&buf[IDX(lny - 1, 0, lnx)], row_sz, MPI_FLOAT, south, 1, comm, &req[1]);

    MPI_Isend(&buf[IDX(1, 0, lnx)], row_sz, MPI_FLOAT, north, 1, comm, &req[2]);
    MPI_Isend(&buf[IDX(lny - 2, 0, lnx)], row_sz, MPI_FLOAT, south, 0, comm, &req[3]);

    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    assert(size <= ny - 2);             


    int dims[1]{size}, periods[1]{0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart);

    int north, south;
    MPI_Cart_shift(cart, 0, +1, &north, &south);  


    const int base = (ny - 2) / size;
    const int rem  = (ny - 2) % size;
    int local_real_ny = base + (rank < rem ? 1 : 0);

    const int lnx = nx;
    const int lny = local_real_ny + 2;     // rows + 2

    vector<float> u (lny * lnx, 0.f),  v (lny * lnx, 0.f),
                  p (lny * lnx, 0.f),  b (lny * lnx, 0.f),
                  un(lny * lnx, 0.f),  vn(lny * lnx, 0.f), pn(lny * lnx, 0.f);

    //Start the Calculation Iteration
    for (int n = 0; n < nt; ++n)
    {
        //Synchronizing u and V
        exchange_ghost(u, lny, lnx, north, south, cart);
        exchange_ghost(v, lny, lnx, north, south, cart);

        for (int j = 1; j <= local_real_ny; ++j)
            for (int i = 1; i < nx - 1; ++i)
            {
                //1 Compute b[j][i]
                const double dudx = (u[IDX(j,i+1,lnx)] - u[IDX(j,i-1,lnx)]) / (2.0*dx);
                const double dvdy = (v[IDX(j+1,i,lnx)] - v[IDX(j-1,i,lnx)]) / (2.0*dy);
                const double dudy = (u[IDX(j+1,i,lnx)] - u[IDX(j-1,i,lnx)]) / (2.0*dy);
                const double dvdx = (v[IDX(j,i+1,lnx)] - v[IDX(j,i-1,lnx)]) / (2.0*dx);
                const double term1 = (1.0/dt)*(dudx + dvdy);
                b[IDX(j,i,lnx)] = static_cast<float>(rho*(term1 - dudx*dudx - 2.0*dudy*dvdx - dvdy*dvdy));
            }

        //2 Compute p[j][i]
        for (int it = 0; it < nit; ++it)
        {
            pn = p;
            exchange_ghost(pn, lny, lnx, north, south, cart);

            for (int j = 1; j <= local_real_ny; ++j)
                for (int i = 1; i < nx - 1; ++i)
                    p[IDX(j,i,lnx)] = static_cast<float>(
                        (dy*dy*(pn[IDX(j,i+1,lnx)] + pn[IDX(j,i-1,lnx)]) +
                         dx*dx*(pn[IDX(j+1,i,lnx)] + pn[IDX(j-1,i,lnx)]) -
                         b[IDX(j,i,lnx)]*dx*dx*dy*dy) /
                        (2.0*(dx*dx + dy*dy)));

            //3 Compute p[j][0] and p[j][nx-1]
            for (int j = 1; j <= local_real_ny; ++j) {
                p[IDX(j,0,lnx)]    = p[IDX(j,1,lnx)];
                p[IDX(j,nx-1,lnx)] = p[IDX(j,nx-2,lnx)];
            }
            if (rank == 0)
                for (int i = 0; i < nx; ++i) p[IDX(0,i,lnx)] = p[IDX(1,i,lnx)];
            if (rank == size-1)
                for (int i = 0; i < nx; ++i) p[IDX(lny-1,i,lnx)] = 0.f;
        }

        //4 Compute p[0][i] and p[ny-1][i]
        un = u;  vn = v;
        exchange_ghost(un, lny, lnx, north, south, cart);
        exchange_ghost(vn, lny, lnx, north, south, cart);
        exchange_ghost(p , lny, lnx, north, south, cart);

        //5 Compute u[j][i] and v[j][i]
        for (int j = 1; j <= local_real_ny; ++j)
            for (int i = 1; i < nx - 1; ++i)
            {
                const double unji = un[IDX(j,i,lnx)];
                const double vnji = vn[IDX(j,i,lnx)];

                const double conv_u_x = unji*dt/dx*(unji - un[IDX(j,i-1,lnx)]);
                const double conv_u_y = unji*dt/dy*(unji - un[IDX(j-1,i,lnx)]);
                const double pres_u = dt/(2.0*rho*dx)*(p[IDX(j,i+1,lnx)] - p[IDX(j,i-1,lnx)]);
                const double diff_u_x = nu*dt/(dx*dx)*(un[IDX(j,i+1,lnx)] - 2.0*unji + un[IDX(j,i-1,lnx)]);
                const double diff_u_y = nu*dt/(dy*dy)*(un[IDX(j+1,i,lnx)] - 2.0*unji + un[IDX(j-1,i,lnx)]);

                const double conv_v_x = vnji*dt/dx*(vnji - vn[IDX(j,i-1,lnx)]);
                const double conv_v_y = vnji*dt/dy*(vnji - vn[IDX(j-1,i,lnx)]);
                const double pres_v = dt/(2.0*rho*dx)*(p[IDX(j+1,i,lnx)] - p[IDX(j-1,i,lnx)]);
                const double diff_v_x = nu*dt/(dx*dx)*(vn[IDX(j,i+1,lnx)] - 2.0*vnji + vn[IDX(j,i-1,lnx)]);
                const double diff_v_y = nu*dt/(dy*dy)*(vn[IDX(j+1,i,lnx)] - 2.0*vnji + vn[IDX(j-1,i,lnx)]);

                u[IDX(j,i,lnx)] = static_cast<float>(unji - conv_u_x - conv_u_y - pres_u + diff_u_x + diff_u_y);
                v[IDX(j,i,lnx)] = static_cast<float>(vnji - conv_v_x - conv_v_y - pres_v + diff_v_x + diff_v_y);
            }

        //6 Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
        for (int j = 1; j <= local_real_ny; ++j) {
            u[IDX(j,0,lnx)]    = 0.f; v[IDX(j,0,lnx)]    = 0.f;
            u[IDX(j,nx-1,lnx)] = 0.f; v[IDX(j,nx-1,lnx)] = 0.f;
        }
        if (rank == 0)
            for (int i = 0; i < nx; ++i) { u[IDX(0,i,lnx)] = 0.f; v[IDX(0,i,lnx)] = 0.f; }
        if (rank == size-1)
            for (int i = 0; i < nx; ++i) { u[IDX(lny-1,i,lnx)] = 1.f; v[IDX(lny-1,i,lnx)] = 0.f; }

        //Synchronizing u and V
        exchange_ghost(u, lny, lnx, north, south, cart);
        exchange_ghost(v, lny, lnx, north, south, cart);
    } 
    //Stop the Calculation Interation 

    //rank0 to obtain the value of calculation
    const int slab_elems = local_real_ny * nx;
    vector<float> send_u(slab_elems), send_v(slab_elems), send_p(slab_elems);
    for (int j = 1; j <= local_real_ny; ++j) {
        std::copy(&u[IDX(j,0,lnx)], &u[IDX(j,0,lnx)] + nx, &send_u[(j-1)*nx]);
        std::copy(&v[IDX(j,0,lnx)], &v[IDX(j,0,lnx)] + nx, &send_v[(j-1)*nx]);
        std::copy(&p[IDX(j,0,lnx)], &p[IDX(j,0,lnx)] + nx, &send_p[(j-1)*nx]);
    }
    vector<int> recvcounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        int rows = base + (r < rem ? 1 : 0);
        recvcounts[r] = rows * nx;
        displs[r]     = (r == 0 ? 0 : displs[r-1] + recvcounts[r-1]);
    }

    vector<float> u_all, v_all, p_all;
    if (rank == 0) {
        u_all.resize(nx*(ny-2));
        v_all.resize(nx*(ny-2));
        p_all.resize(nx*(ny-2));
    }
    MPI_Gatherv(send_u.data(), slab_elems, MPI_FLOAT,
                u_all.data(), recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(send_v.data(), slab_elems, MPI_FLOAT,
                v_all.data(), recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(send_p.data(), slab_elems, MPI_FLOAT,
                p_all.data(), recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        ofstream uf("u.dat"), vf("v.dat"), pf("p.dat");
        for (int j = 1, offset = 0; j < ny-1; ++j, offset += nx) {
            for (int i = 0; i < nx; ++i) uf << u_all[offset+i] << ' ';
            uf << '\n';
            for (int i = 0; i < nx; ++i) vf << v_all[offset+i] << ' ';
            vf << '\n';
            for (int i = 0; i < nx; ++i) pf << p_all[offset+i] << ' ';
            pf << '\n';
        }
    }

    MPI_Finalize();
    return 0;
}
