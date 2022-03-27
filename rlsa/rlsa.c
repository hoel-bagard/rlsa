#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <math.h>

/**
 * Apply the RLS algorithm horizontally on the given image.
 * This function eliminates horizontal white runs whose lengths are smaller than the given value.
 *
 * Args:
 *     img: Image on which to apply the rlsa horizontal process (gets modified).
 *     dims: Number of rows and columns in the image.
 *     hsv: The treshold smoothing value.
 */
static void rlsa_horizontal(uint8_t* img, long int rows, long int cols, int hsv) {
  for(int i = 0; i < rows; i++) {
    int last_black_px = 0;  // Index of the last 0 found
    for(int j = 0; j < cols; j++) {
      if (img[i*cols + j] == 0) {
        if (j-last_black_px <= hsv && last_black_px != 0)   // last_black_px != 0 is to avoid linking borders to the text.
          for(int k = last_black_px; k < j; k++)
            img[i*cols + k] = 0;
        last_black_px = j;
      }
    }
  }
}

/**
 * Same as above, but vertically.
 */
static void rlsa_vertical(uint8_t* img, long int rows, long int cols, int vsv) {
  for(int j = 0; j < cols; j++) {
    int last_black_px = 0;
    for(int i = 0; i < rows; i++)
      if (img[i*cols + j] == 0) {
        if (i-last_black_px <= vsv && last_black_px != 0)
          for(int k = last_black_px; k < i; k++)
            img[k*cols + j] = 0;
        last_black_px = i;
      }
  }
}

/**
 * Apply the Run Length Smoothing Algorithm on the given image.
 *
 * Args:
 *     img: Image on which to apply the rlsa process (gets modified).
 *     dims: Number of rows and columns in the image.
 *     hsv: The horizontal treshold smoothing value.
 *     vsv: The vertical treshold smoothing value.
 */
static void rlsa(uint8_t* img, npy_intp* dims, int hsv, int vsv, int ahsv ) {
  long int rows = dims[0];
  long int cols = dims[1];

  uint8_t horizontal_rlsa_img[rows * cols];
  memcpy(horizontal_rlsa_img, img, sizeof(horizontal_rlsa_img));
  rlsa_horizontal(horizontal_rlsa_img, rows, cols, hsv);
  rlsa_vertical(img, rows, cols, vsv);

  // And operation between the vetical and horizontal results.
  for(int i = 0; i < rows; i++)
    for(int j = 0; j < cols; j++)
      if (img[i*cols + j] == 0 || horizontal_rlsa_img[i*cols + j] == 0)
        img[i*cols + j] = 0;

  rlsa_horizontal(img, rows, cols, ahsv);
}


static PyObject *rlsa_wrapper(PyObject *self, PyObject *args) {
  import_array();
  import_umath();

  PyArrayObject* in_img = NULL;
  int vsv, hsv, ahsv;

  if (!PyArg_ParseTuple(args, "Oiii", &in_img, &hsv, &vsv, &ahsv))
    return NULL;

  in_img = (PyArrayObject*) PyArray_Cast(in_img, NPY_UINT8);

  int nb_dims = PyArray_NDIM(in_img);  // number of dimensions
  if (nb_dims != 2) PyErr_SetString(PyExc_ValueError, "Numpy array must be 2D.");
  npy_intp* dims = PyArray_DIMS(in_img);  // npy_intp array of length nb_dims showing length in each dim.
  uint8_t* in_data = (uint8_t*)PyArray_DATA(in_img);  // Pointer to data.

  // Copy the input image data to an output image (that we will modify from now on).
  uint8_t* out_data = (uint8_t*)malloc(dims[0] * dims[1] * sizeof(uint8_t));  // uint8_t out_data[dims[0] * dims[1]];
  memcpy(out_data, in_data, dims[0] * dims[1] * sizeof(uint8_t));

  rlsa(out_data, dims, hsv, vsv, ahsv);

  // create a python numpy array from the out array
  PyArrayObject* out_img = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_UINT8, out_data);

  /* Py_DECREF(in_img); */

  return PyArray_Return(out_img);
}


static PyObject *rlsa_wrapper_horizontal(PyObject *self, PyObject *args) {
  import_array();
  import_umath();

  PyArrayObject* in_img = NULL;
  int hsv;

  if (!PyArg_ParseTuple(args, "Oi", &in_img, &hsv))
    return NULL;

  in_img = (PyArrayObject*) PyArray_Cast(in_img, NPY_UINT8);

  int nb_dims = PyArray_NDIM(in_img);  // number of dimensions
  if (nb_dims != 2) PyErr_SetString(PyExc_ValueError, "Numpy array must be 2D.");
  npy_intp* dims = PyArray_DIMS(in_img);  // npy_intp array of length nb_dims showing length in each dim.
  uint8_t* in_data = (uint8_t*)PyArray_DATA(in_img);  // Pointer to data.

  // Copy the input image data to an output image (that we will modify from now on).
  uint8_t* out_data = (uint8_t*)malloc(dims[0] * dims[1] * sizeof(uint8_t));  // uint8_t out_data[dims[0] * dims[1]];
  memcpy(out_data, in_data, dims[0] * dims[1] * sizeof(uint8_t));

  rlsa_horizontal(out_data, dims[0], dims[1], hsv);

  // create a python numpy array from the out array
  PyArrayObject* out_img = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_UINT8, out_data);

  return PyArray_Return(out_img);
}


static PyObject *rlsa_wrapper_vertical(PyObject *self, PyObject *args) {
  import_array();
  import_umath();

  PyArrayObject* in_img = NULL;
  int vsv;

  if (!PyArg_ParseTuple(args, "Oi", &in_img, &vsv))
    return NULL;

  in_img = (PyArrayObject*) PyArray_Cast(in_img, NPY_UINT8);

  int nb_dims = PyArray_NDIM(in_img);  // number of dimensions
  if (nb_dims != 2) PyErr_SetString(PyExc_ValueError, "Numpy array must be 2D.");
  npy_intp* dims = PyArray_DIMS(in_img);  // npy_intp array of length nb_dims showing length in each dim.
  uint8_t* in_data = (uint8_t*)PyArray_DATA(in_img);  // Pointer to data.

  // Copy the input image data to an output image (that we will modify from now on).
  uint8_t* out_data = (uint8_t*)malloc(dims[0] * dims[1] * sizeof(uint8_t));  // uint8_t out_data[dims[0] * dims[1]];
  memcpy(out_data, in_data, dims[0] * dims[1] * sizeof(uint8_t));

  rlsa_vertical(out_data, dims[0], dims[1], vsv);

  // create a python numpy array from the out array
  PyArrayObject* out_img = (PyArrayObject*) PyArray_SimpleNewFromData(2, dims, NPY_UINT8, out_data);

  return PyArray_Return(out_img);
}


PyDoc_STRVAR(rlsa_doc,
             "rlsa(binary_img, hsv, vsv, ahsv, /)\n"
             "--\n\n"
             "Applies the Run Length Smoothing Algorithm on an image.\n"
             "\n"
             "Args:\n"
             "    binary_img (npt.NDArray[np.uint8]): The black and white input image.\n"
             "    hsv (int): The horizontal threshold, i.e., the number of white pixels needed to 'separate' two black pixels.\n"
             "    vsv (int): The vertical threshold.\n"
             "    ahsv (int): Second horizontal threshold, for the (optional) second horizontal pass.\n"
             "\n"
             "Returns:\n"
             "    out_img (npt.NDArray[np.uint8]): New image with the rlsa applied on it.\n"
             "\n"
             "Example:\n"
             "    >>> img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)\n"
             "    >>> _, binary_img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)\n"
             "    >>> out_img = rlsa(binary_img, 25, 25, 3)\n"
             );

PyDoc_STRVAR(rlsa_horizontal_doc,
             "rlsa_horizontal(binary_img, hsv, /)\n"
             "--\n\n"
             "Applies the horizontal component of RLSA on an image.\n"
             "\n"
             "Args:\n"
             "    binary_img (npt.NDArray[np.uint8]): The black and white input image.\n"
             "    hsv (int): The horizontal threshold, i.e., the number of white pixels needed to 'separate' two black pixels.\n"
             "\n"
             "Returns:\n"
             "    out_img (npt.NDArray[np.uint8]): New image with the horizontal rlsa applied on it.\n"
             "\n"
             "Example:\n"
             "    >>> img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)\n"
             "    >>> _, binary_img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)\n"
             "    >>> out_img = rlsa_horizontal(binary_img, 25)\n"
             );

PyDoc_STRVAR(rlsa_vertical_doc,
             "rlsa_vertical(binary_img, vsv, /)\n"
             "--\n\n"
             "Applies the vertical component of RLSA on an image.\n"
             "\n"
             "Args:\n"
             "    binary_img (npt.NDArray[np.uint8]): The black and white input image.\n"
             "    vsv (int): The vertical threshold, i.e., the number of white pixels needed to 'separate' two black pixels.\n"
             "\n"
             "Returns:\n"
             "    out_img (npt.NDArray[np.uint8]): New image with the vertical rlsa applied on it.\n"
             "\n"
             "Example:\n"
             "    >>> img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)\n"
             "    >>> _, binary_img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)\n"
             "    >>> out_img = rlsa_vertical(binary_img, 25)\n"
             );

static PyMethodDef RLSAMethods[] = {
  {"rlsa",  rlsa_wrapper, METH_VARARGS, rlsa_doc},
  {"rlsa_horizontal",  rlsa_wrapper_horizontal, METH_VARARGS, rlsa_horizontal_doc},
  {"rlsa_vertical",  rlsa_wrapper_vertical, METH_VARARGS, rlsa_vertical_doc},
  {NULL, NULL, 0, NULL}  /* Sentinel */
};


static struct PyModuleDef rlsa_module = {
  PyModuleDef_HEAD_INIT,
  "rlsa",   /* Name of module */
  "Run Length Smoothing Algorithm package.",  // Module description.
  -1,       /* Size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
  RLSAMethods
};


PyMODINIT_FUNC PyInit_rlsa(void) {
  return PyModule_Create(&rlsa_module);
}
