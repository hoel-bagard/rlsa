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
    PyObject *arg1=NULL, *out=NULL;
    PyObject *arr1=NULL, *oarr=NULL;
    int vsv, hsv;

    import_array();
    import_umath();

    printf("1\n");
    if (!PyArg_ParseTuple(args, "iii", &arg1, &vsv, hsv))
      return NULL;
    printf("2\n");

    arr1 = PyArray_FROM_OTF(arg1, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (arr1 == NULL) return NULL;
#if NPY_API_VERSION >= 0x0000000c
    oarr = PyArray_FROM_OTF(out, NPY_UINT8, NPY_ARRAY_INOUT_ARRAY2);
#else
    oarr = PyArray_FROM_OTF(out, NPY_UINT8, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (oarr == NULL) goto fail;

    int nd = PyArray_NDIM(arr1);
    printf("nd: %d", nd);

    /* code that makes use of arguments */
    /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                     showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

       If an error occurs goto fail.
     */

    Py_DECREF(arr1);
#if NPY_API_VERSION >= 0x0000000c
    PyArray_ResolveWritebackIfCopy(oarr);
#endif
    Py_DECREF(oarr);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(arr1);
#if NPY_API_VERSION >= 0x0000000c
    PyArray_DiscardWritebackIfCopy(oarr);
#endif
    Py_XDECREF(oarr);
    return NULL;
}




static PyMethodDef RLSAMethods[] = {
  {"rlsa",  rlsa_wrapper, METH_VARARGS, "Run Length Smoothing Algorithm."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef rlsa_module = {
  PyModuleDef_HEAD_INIT,
  "rlsa",   /* name of module */
  NULL, /* module documentation, may be NULL */
  -1,       /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
  RLSAMethods
};


PyMODINIT_FUNC PyInit_rlsa(void) {
  return PyModule_Create(&rlsa_module);
}
