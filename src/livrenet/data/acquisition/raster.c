#include <stdint.h>
#include <math.h>
#include <stddef.h>

/* Same face order, backface/near rejection and perspective z-buffer as the
   Python reference. No fast-math, approximate clipping, GPU or lower resolution. */
void raster_kernel(const double *camera_points, const int64_t *faces,
                   const double *inc, const double *areas, int64_t nf,
                   int w, int h, double fx, double fy, double cx, double cy,
                   double near, double *depth, int64_t *face_map,
                   float *bary, float *incidence, double *weight, double *projected) {
    for (int64_t i=0; i<(int64_t)w*h; ++i) { depth[i]=INFINITY; face_map[i]=-1; }
    for (int64_t f=0; f<nf; ++f) {
        double z[3], x[3], y[3];
        int valid=inc[f]>0.;
        for (int k=0; k<3; ++k) {
            const double *p=camera_points+3*faces[3*f+k];
            z[k]=p[2];
            if (z[k]<=near) valid=0;
            x[k]=fx*p[0]/z[k]+cx; y[k]=fy*p[1]/z[k]+cy;
        }
        if (!valid) continue;
        double area=.5*fabs((x[1]-x[0])*(y[2]-y[0])-(y[1]-y[0])*(x[2]-x[0]));
        if (area<1e-10) continue;
        projected[f]=area;
        double xminf=floor(fmin(x[0],fmin(x[1],x[2]))), xmaxf=ceil(fmax(x[0],fmax(x[1],x[2])));
        double yminf=floor(fmin(y[0],fmin(y[1],y[2]))), ymaxf=ceil(fmax(y[0],fmax(y[1],y[2])));
        if (xminf>w-1 || xmaxf<0 || yminf>h-1 || ymaxf<0) continue;
        int xmin=(int)fmax(0,xminf), xmax=(int)fmin(w-1,xmaxf);
        int ymin=(int)fmax(0,yminf), ymax=(int)fmin(h-1,ymaxf);
        double denom=(y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2]);
        if (fabs(denom)<1e-12) continue;
        for (int iy=ymin; iy<=ymax; ++iy) for (int ix=xmin; ix<=xmax; ++ix) {
            double px=ix+.5, py=iy+.5;
            double b0=((y[1]-y[2])*(px-x[2])+(x[2]-x[1])*(py-y[2]))/denom;
            double b1=((y[2]-y[0])*(px-x[2])+(x[0]-x[2])*(py-y[2]))/denom;
            double b2=1.-b0-b1;
            if (b0 < -1e-7 || b1 < -1e-7 || b2 < -1e-7) continue;
            double a0=b0/z[0], a1=b1/z[1], a2=b2/z[2];
            double inverse=a0+a1+a2;
            double d=1./fmax(inverse,1e-12);
            size_t i=(size_t)iy*w+ix;
            if (d>=depth[i]) continue;
            depth[i]=d; face_map[i]=f;
            bary[3*i]=(float)(a0/fmax(inverse,1e-12));
            bary[3*i+1]=(float)(a1/fmax(inverse,1e-12));
            bary[3*i+2]=(float)(a2/fmax(inverse,1e-12));
            incidence[i]=(float)inc[f]; weight[i]=areas[f]/area;
        }
    }
    for (int64_t i=0; i<(int64_t)w*h; ++i) if (!isfinite(depth[i])) depth[i]=0.;
}
