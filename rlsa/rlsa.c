#define PY_SSIZE_T_CLEAN
#include <Python.h>


int interval_counter(int start, int end) {
  int sum = 0;
  for (int i = start; i < end; i++) sum++;  // For loop to be able to time it
  return sum;
}


static PyObject *interval_counter_wrapper(PyObject *self, PyObject *args) {
  int start, end;
  int sum;

  if (!PyArg_ParseTuple(args, "ii", &start, &end))
    return NULL;

  sum = interval_counter(start, end);

  return PyLong_FromLong(sum);
}


static PyMethodDef CounterMethods[] = {
  {"interval_counter",  interval_counter_wrapper, METH_VARARGS, "Returns number of numbers in the interval."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef mycounter_module = {
  PyModuleDef_HEAD_INIT,
  "mycounter",   /* name of module */
  NULL, /* module documentation, may be NULL */
  -1,       /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
  CounterMethods
};


PyMODINIT_FUNC PyInit_mycounter(void) {
    return PyModule_Create(&mycounter_module);
}
