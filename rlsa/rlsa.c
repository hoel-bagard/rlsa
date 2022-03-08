#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <math.h>


static void rlsa(char **args, npy_intp *dimensions, npy_intp* steps, void* data) {
  npy_intp i;
  npy_intp n = dimensions[0];
  char *in = args[0], *out = args[1];
  npy_intp in_step = steps[0], out_step = steps[1];

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

static PyMethodDef RLSAMethods[] = {
  {NULL, NULL, 0, NULL}
};



PyUFuncGenericFunction funcs[1] = {&rlsa};  /*This a pointer to the above function*/
static void *data[1] = {NULL};
static char types[2] = {NPY_DOUBLE, NPY_DOUBLE};  /* These are the input and return dtypes of the function.*/

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "rlsa", /* name of module */
  NULL,   /* module documentation, may be NULL */
  -1,     /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
  RLSAMethods
};

PyMODINIT_FUNC PyInit_rlsa(void) {
  PyObject *m, *rlsa, *d;
  m = PyModule_Create(&moduledef);
  if (!m)
    return NULL;

  import_array();
  import_umath();

  rlsa = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                 PyUFunc_None, "rlsa",
                                 "Run Length Smoothing Algorithm", 0);

  d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "rlsa", rlsa);
  Py_DECREF(rlsa);

  return m;
}
