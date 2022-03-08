#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <math.h>
#include <stdio.h>


static void rlsa(char **args, npy_intp *dimensions, npy_intp* steps, void* data) {
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in = args[0], *out = args[1];
  npy_intp in_step = steps[0], out_step = steps[1];

  int a = dimensions[0];
  printf("n: %d\n", a);

  double tmp;

  for (i = 0; i < n; i++) {
    /*BEGIN main ufunc computation*/
    tmp = *(double *)in;
    tmp /= 1-tmp;
    *((double *)out) = log(tmp);
    /*END main ufunc computation*/

    in += in_step;
    out += out_step;
  }
}

static PyObject *rlsa_wrapper(PyObject *self, PyObject *args) {
  int debug = 1;
  PyArrayObject* in_img = NULL;
  int* out_data = NULL;

  /* PyObject *out=NULL, *oarr=NULL; */
  int vsv, hsv;

  import_array();
  import_umath();

  if (!PyArg_ParseTuple(args, "Oii", &in_img, &vsv, &hsv))
    return NULL;

  // Needs to be int32 (even if the image is in uint8), because (I think) C ints are usually 32bits nowadays.
  in_img = (PyArrayObject*) PyArray_Cast(in_img, NPY_INT32);

  int nb_dims = PyArray_NDIM(in_img);  // number of dimensions
  npy_intp* dims = PyArray_DIMS(in_img);  // npy_intp array of length nb_dims showing length in each dim.
  int* in_data = (int*)PyArray_DATA(in_img);  // Pointer to data.
  if (debug) {
    printf("Received array with %d dimensions\n", nb_dims);
    printf("First dimension has %ld elements, second one has %ld elements\n", dims[0], dims[1]);
    printf("First int is %d\n", in_data[0]);
  }

  out_data = in_data;
  // create a python numpy array from the out array
  PyArrayObject* output = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_UINT8, (void*)out_data);
  return PyArray_Return(output);

  /* If an error occurs goto fail. */

  /* Py_DECREF(arr); */
  /* Py_INCREF(Py_None); */
  /* return Py_None; */

 /* fail: */
 /*  Py_XDECREF(arr); */
 /*  return NULL; */
}


static PyMethodDef RLSAMethods[] = {
  {"rlsa",  rlsa_wrapper, METH_VARARGS, "Applies the Run Length Smoothing Algorithm on an image."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef rlsa_module = {
  PyModuleDef_HEAD_INIT,
  "rlsa",   /* name of module */
  "Run Length Smoothing Algorithm package.",
  -1,       /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
  RLSAMethods
};


PyMODINIT_FUNC PyInit_rlsa(void) {
  return PyModule_Create(&rlsa_module);
}
