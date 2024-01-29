#include "mex.h"

#define Z plhs[0]
#define X prhs[0]
#define Y prhs[1]
#define S prhs[2]
#define pi(x) printf("%d ",x)
#define pf(x) printf("%f ",x)
#define min(x,y) (((x)<(y)) ? (x) : (y))
#define max(x,y) (((x)<(y)) ? (y) : (x))
#define pd(d) printf("%d\n",d)

int *GetDimensions(const mxArray *A, int numdims) {
  int nd, i;
  const int *dims0; 
  int *dims;

  nd = mxGetNumberOfDimensions(A);
  numdims = max(nd,numdims);
  dims = mxMalloc(sizeof(int)*numdims);
  dims0 = mxGetDimensions(A);
  for (i=0;  i<nd;      i++) dims[i] = dims0[i];
  for (i=nd; i<numdims; i++) dims[i] = 1;
  return dims;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]) {

  int i;
  bool xcomplex, ycomplex;
  int xnd, *xdims, *xskip, *xbac1;
  int ynd, *ydims, *yskip, *ybac1;
  int znd, *zdims, *zi;
  int snd;
  double *xr, *yr, *zr, s;
  double *xc, *yc, *zc, t;

  if (nrhs != 3) mexErrMsgTxt("Exactly 3 arguments required.");
  xcomplex = mxIsComplex(X);
  ycomplex = mxIsComplex(Y);

  xnd = mxGetNumberOfDimensions(X);
  ynd = mxGetNumberOfDimensions(Y);
  znd = max(xnd,ynd)+1;
  snd = min(*mxGetPr(S),max(xnd,ynd));

  xdims = GetDimensions(X,znd);
  ydims = GetDimensions(Y,znd);
  zdims = mxMalloc(sizeof(int)*znd);
  for (i=0; i<znd; i++) {
   if      (xdims[i]==1)        zdims[i] = ydims[i];
   else if (ydims[i]==1)        zdims[i] = xdims[i];
   else if (xdims[i]==ydims[i]) zdims[i] = xdims[i];
   else mexErrMsgTxt("Array dimensions do not match.");
  }
 
  xskip = mxMalloc(sizeof(int)*znd);
  yskip = mxMalloc(sizeof(int)*znd);
  xbac1 = mxMalloc(sizeof(int)*znd);
  ybac1 = mxMalloc(sizeof(int)*znd);
  xskip[0] = 1;
  yskip[0] = 1;
  for (i=1; i<znd; i++) {
    xskip[i]   = xskip[i-1]*xdims[i-1]--;
    yskip[i]   = yskip[i-1]*ydims[i-1]--;
    xbac1[i-1] = xskip[i-1]*xdims[i-1];
    ybac1[i-1] = yskip[i-1]*ydims[i-1];
  }
  xbac1[znd-1] = xskip[znd-1]*(--xdims[znd-1]);
  ybac1[znd-1] = yskip[znd-1]*(--ydims[znd-1]);
  for (i=0; i<znd-1; i++) {
    xskip[i] = xskip[i]*(xdims[i]!=0);
    yskip[i] = yskip[i]*(ydims[i]!=0);
    xbac1[i] = xbac1[i]*(xdims[i]!=0);
    ybac1[i] = ybac1[i]*(ydims[i]!=0);
  };

  zi = mxMalloc(sizeof(int)*znd);
  for (i=0; i<znd; i++) {
    zi[i] = 1;
  }
  zi[znd-1] = 0;
  
  if (mxIsEmpty(X) || mxIsEmpty(Y)) {
    Z = mxCreateNumericArray(znd-snd,zdims+snd,mxDOUBLE_CLASS,mxREAL);

  } else if (!xcomplex && !ycomplex) {
    Z  = mxCreateNumericArray(znd-snd,zdims+snd,mxDOUBLE_CLASS,mxREAL);
    zr = mxGetPr(Z);
    xr    = mxGetPr(X);
    yr    = mxGetPr(Y);

    s = 0.0; 
    while(zi[znd-1]==0) {
      s += *xr * *yr;
      for (i=0; i<znd; i++) {
        if (zi[i]==zdims[i]) {
          zi[i] = 1;
          xr -= xbac1[i];
          yr -= ybac1[i];
        } else {
          zi[i] ++;
          xr += xskip[i];
          yr += yskip[i];
          if (i>=snd) {
            *(zr++) = s;
            s       = 0.0;
          }
          break;
        }
      }
    }

  } else if (!xcomplex &&  ycomplex) {
    Z  = mxCreateNumericArray(znd-snd,zdims+snd,mxDOUBLE_CLASS,mxCOMPLEX);
    zr = mxGetPr(Z); zc = mxGetPi(Z);
    xr = mxGetPr(X);
    yr = mxGetPr(Y); yc = mxGetPi(Y);

    s = 0.0; t = 0.0;
    while(zi[znd-1]==0) {
      s += *xr * *yr;
      t += *xr * *yc;
      for (i=0; i<znd; i++) {
        if (zi[i]==zdims[i]) {
          zi[i] = 1;
          xr -= xbac1[i];
          yr -= ybac1[i]; yc -= ybac1[i];
        } else {
          zi[i] ++;
          xr += xskip[i];
          yr += yskip[i]; yc += yskip[i];
          if (i>=snd) {
            *(zr++) = s;   *(zc++) = t;
            s       = 0.0; t       = 0.0;
          }
          break;
        }
      }
    }
  } else if ( xcomplex && !ycomplex) {
    Z  = mxCreateNumericArray(znd-snd,zdims+snd,mxDOUBLE_CLASS,mxCOMPLEX);
    zr = mxGetPr(Z); zc = mxGetPi(Z);
    xr = mxGetPr(X); xc = mxGetPi(X);
    yr = mxGetPr(Y);

    s = 0.0; t = 0.0;
    while(zi[znd-1]==0) {
      s += *xr * *yr;
      t += *xc * *yr;
      for (i=0; i<znd; i++) {
        if (zi[i]==zdims[i]) {
          zi[i] = 1;
          xr -= xbac1[i]; xc -= xbac1[i];
          yr -= ybac1[i];
        } else {
          zi[i] ++;
          xr += xskip[i]; xc += xskip[i];
          yr += yskip[i];
          if (i>=snd) {
            *(zr++) = s;   *(zc++) = t;
            s       = 0.0; t       = 0.0;
          }
          break;
        }
      }
    }
  } else /* ( xcomplex &&  ycomplex) */ {
    Z  = mxCreateNumericArray(znd-snd,zdims+snd,mxDOUBLE_CLASS,mxCOMPLEX);
    zr = mxGetPr(Z); zc = mxGetPi(Z);
    xr = mxGetPr(X); xc = mxGetPi(X);
    yr = mxGetPr(Y); yc = mxGetPi(Y);

    s = 0.0; t = 0.0;
    while(zi[znd-1]==0) {
      s += *xr * *yr - *xc * *yc;
      t += *xc * *yr + *xr * *yc;
      for (i=0; i<znd; i++) {
        if (zi[i]==zdims[i]) {
          zi[i] = 1;
          xr -= xbac1[i]; xc -= xbac1[i];
          yr -= ybac1[i]; yc -= ybac1[i];
        } else {
          zi[i] ++;
          xr += xskip[i]; xc += xskip[i];
          yr += yskip[i]; yc += yskip[i];
          if (i>=snd) {
            *(zr++) = s;   *(zc++) = t;
            s       = 0.0; t       = 0.0;
          }
          break;
        }
      }
    }
  }

  mxFree(xdims); mxFree(xskip); mxFree(xbac1);
  mxFree(ydims); mxFree(yskip); mxFree(ybac1);
  mxFree(zdims); mxFree(zi);

}
    

    
