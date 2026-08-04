use core::ffi::c_int;

use pyo3::ffi::PyFrameObject;
use pyo3::ffi::PyInterpreterState;
use pyo3::ffi::PyObject;
use pyo3::ffi::PyThreadState;

pub type _PyFrameEvalFunction =
    unsafe extern "C" fn(*mut PyThreadState, *mut PyFrameObject, c_int) -> *mut pyo3::ffi::PyObject;

extern "C" {
    pub fn _PyEval_EvalFrameDefault(
        ts: *mut PyThreadState,
        frame: *mut PyFrameObject,
        extra: c_int,
    ) -> *mut PyObject;
}

unsafe fn python_symbol<T: Copy>(name: &'static [u8]) -> T {
    let symbol = libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr().cast());
    assert!(!symbol.is_null(), "missing CPython symbol");
    std::mem::transmute_copy(&symbol)
}

/// Resolve these private APIs lazily: Python 3.8 does not export them, while
/// abi3 extensions are still loaded and validated against its symbol table.
#[allow(non_snake_case)]
pub unsafe fn _PyInterpreterState_GetEvalFrameFunc(
    interp: *mut PyInterpreterState,
) -> _PyFrameEvalFunction {
    let function: unsafe extern "C" fn(*mut PyInterpreterState) -> _PyFrameEvalFunction =
        python_symbol(b"_PyInterpreterState_GetEvalFrameFunc\0");
    function(interp)
}

#[allow(non_snake_case)]
pub unsafe fn _PyInterpreterState_SetEvalFrameFunc(
    interp: *mut PyInterpreterState,
    eval_frame: _PyFrameEvalFunction,
) {
    let function: unsafe extern "C" fn(*mut PyInterpreterState, _PyFrameEvalFunction) =
        python_symbol(b"_PyInterpreterState_SetEvalFrameFunc\0");
    function(interp, eval_frame)
}

#[allow(non_snake_case)]
pub unsafe fn PyInterpreterState_Get() -> *mut PyInterpreterState {
    let function: unsafe extern "C" fn() -> *mut PyInterpreterState =
        python_symbol(b"PyInterpreterState_Get\0");
    function()
}
