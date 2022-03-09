#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <math.h>
#include <stdio.h>


/**
 * Apply the RLS algorithm horizontally on the given image.
 * This function eliminates horizontal white runs whose lengths are smaller than the given value.
 *
 * Note: This function can be used to do the operation vertically by simply passing the transpose.
 *
 * Args:
 *     img:
 *     dims: Number of rows and columns in the image.
 *     value:
 */
static void rlsa(int* img, npy_intp* dims, int value) {  // _horizontal
  long int rows = dims[0];
  long int cols = dims[1];

  /* for(int i = 0; i < rows; i++) { */
  /*   for(int j = 0; j < cols; j++) { */
  /*     if (in_img[i*cols + j] != out_img[i*cols + j]) { */
  /*       printf("In: %d\n", in_img[i*cols + j]); */
  /*       printf("Out: %d\n", out_img[i*cols + j]); */
  /*     } */
  /*   } */
  /* } */

  for(int i = 0; i < rows; i++) {
    int count = 0;  // Index of the last 0 found
    for(int j = 0; j < cols; j++) {
      if (img[i*cols + j] == 0) {
        if (j-count <= value)
          for(int k = count; k < j; k++)
            img[i*cols + k] = 0;
        count = j;
      }
    }
  }
}


/* def rlsa(img: np.ndarray, value_horizontal: int, value_vertical: int, ahsv: Optional[int] = None) -> np.ndarray: */
/* """Run Length Smoothing Algorithm. */

/*   Args: */
/* img (np.ndarray): The image to process. */
/*   value_horizontal (int): The horizontal threshold (hsv=300 in the paper) */
/*   value_vertical (int): The vertical threshold (vsv=500 in the paper) */

/*   Returns: */
/* The resulting image. */
/*   """ */
/*   horizontal_rlsa = rlsa_horizontal(img, value_horizontal) */
/*   vertical_rlsa = rlsa_horizontal(img.T, value_vertical).T */
/*   combined_result = cv2.bitwise_and(horizontal_rlsa, vertical_rlsa) */
/*   rlsa_result = rlsa_horizontal(combined_result, ahsv if ahsv else value_horizontal // 10) */
/*                                                                       return rlsa_result */




static PyObject *rlsa_wrapper(PyObject *self, PyObject *args) {
  int debug = 1;
  PyArrayObject* in_img = NULL;
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

  /* out_data = in_data; */

  /* int out_data[dims[0] * dims[1]]; */
  /* memcpy(out_data, in_img, sizeof(out_data)); */
  int* out_data = (int*)malloc(dims[0] * dims[1] * sizeof(int));
  memcpy(out_data, in_data, dims[0] * dims[1] * sizeof(int));
  rlsa(out_data, dims, hsv);

  // create a python numpy array from the out array
  PyArrayObject* out_img = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_INT32, out_data);
  out_img = (PyArrayObject*) PyArray_Cast(out_img, NPY_UINT8);
  return PyArray_Return(out_img);

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
