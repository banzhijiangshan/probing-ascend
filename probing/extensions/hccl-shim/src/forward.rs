//! Lazy forward to the real CANN `libprofapi.so` (never dlopen the shim name).

#![cfg(target_os = "linux")]

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

type ProfCommandHandle = Option<unsafe extern "C" fn(u32, *mut c_void, u32) -> i32>;

type FnRegisterCallback = unsafe extern "C" fn(u32, ProfCommandHandle) -> i32;
type FnRegisterProfileCallback = unsafe extern "C" fn(i32, *mut c_void, u32) -> i32;
type FnRegisterLegacyCallback = unsafe extern "C" fn(*mut c_void) -> i32;
type FnSetProfCommand = unsafe extern "C" fn(*mut c_void, u32) -> i32;
type FnGetModelDevice = unsafe extern "C" fn(u32, *mut u32) -> i32;
type FnSetStepInfo = unsafe extern "C" fn(u64, u16, *mut c_void) -> i32;
type FnRegTypeInfo = unsafe extern "C" fn(u16, u32, *const c_char) -> i32;
type FnInit = unsafe extern "C" fn(u32, *mut c_void, u32) -> i32;
type FnFinalize = unsafe extern "C" fn() -> i32;
type FnReportApi = unsafe extern "C" fn(u32, *const c_void) -> i32;
type FnReportBlob = unsafe extern "C" fn(u32, *const c_void, u32) -> i32;
type FnReportData = unsafe extern "C" fn(u32, u32, *mut c_void, u32) -> i32;
type FnNotifySetDevice = unsafe extern "C" fn(u32, u32, bool) -> i32;
type FnSetConfig = unsafe extern "C" fn(u32, *const c_char, usize) -> i32;
type FnSetModelDevice = unsafe extern "C" fn(u32, u32) -> i32;
type FnStartStop = unsafe extern "C" fn(u32, *const c_void, u32) -> i32;
type FnGetHashId = unsafe extern "C" fn(*const c_char, u32) -> u64;
type FnStr2Id = unsafe extern "C" fn(*const c_char, usize) -> u64;
type FnSysCycleTime = unsafe extern "C" fn() -> u64;

struct RealApi {
    register_callback: FnRegisterCallback,
    register_profile_callback: FnRegisterProfileCallback,
    reg_reporter_callback: FnRegisterLegacyCallback,
    reg_ctrl_callback: FnRegisterLegacyCallback,
    reg_device_callback: FnRegisterLegacyCallback,
    set_prof_command: FnSetProfCommand,
    get_model_device: FnGetModelDevice,
    set_step_info: FnSetStepInfo,
    reg_type_info: FnRegTypeInfo,
    init: FnInit,
    finalize: FnFinalize,
    report_api: FnReportApi,
    report_event: FnReportApi,
    report_compact: FnReportBlob,
    report_additional: FnReportBlob,
    report_data: FnReportData,
    notify_set_device: FnNotifySetDevice,
    set_config: FnSetConfig,
    set_model_device: FnSetModelDevice,
    unset_model_device: FnSetModelDevice,
    start: FnStartStop,
    stop: FnStartStop,
    get_hash_id: FnGetHashId,
    str_to_id: FnStr2Id,
    sys_cycle_time: FnSysCycleTime,
}

struct RealLib {
    handle: *mut c_void,
    api: RealApi,
}

unsafe impl Send for RealLib {}

impl Drop for RealLib {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                libc::dlclose(self.handle);
            }
        }
    }
}

static INIT: Lazy<Mutex<Option<RealLib>>> = Lazy::new(|| Mutex::new(None));
static INIT_FAILED: AtomicBool = AtomicBool::new(false);
static LOGGED_INIT: AtomicBool = AtomicBool::new(false);

const ENV_REAL: &str = "PROBING_HCCL_PROFAPI_REAL";
const REAL_BASENAME: &str = "libprofapi.so.real";
const ENV_ASCEND_HOME: &str = "ASCEND_HOME";
const ENV_ASCEND_HOME_PATH: &str = "ASCEND_HOME_PATH";
const ENV_ASCEND_INSTALL: &str = "ASCEND_INSTALL_PATH";

fn log_once(msg: &str) {
    if std::env::var_os("PROBING_HCCL_SHIM_LOG").is_some()
        && LOGGED_INIT
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        crate::log::info(msg);
    }
}

fn shim_directory() -> Option<PathBuf> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    for line in maps.lines() {
        if !line.contains("libprofapi.so") {
            continue;
        }
        let path = line.split_whitespace().last()?;
        let p = Path::new(path);
        if p.is_absolute() {
            return p.parent().map(|d| d.to_path_buf());
        }
    }
    None
}

fn ascend_lib_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for key in [ENV_ASCEND_HOME, ENV_ASCEND_HOME_PATH, ENV_ASCEND_INSTALL] {
        if let Ok(v) = std::env::var(key) {
            let base = PathBuf::from(v);
            out.push(base.join("lib64"));
            out.push(base.join("lib"));
            out.push(base.join("aarch64-linux/lib64"));
            out.push(base.join("x86_64-linux/lib64"));
        }
    }
    out
}

fn candidate_real_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var(ENV_REAL) {
        out.push(PathBuf::from(p));
    }
    if let Some(dir) = shim_directory() {
        out.push(dir.join(REAL_BASENAME));
    }
    for libdir in ascend_lib_dirs() {
        out.push(libdir.join("libprofapi.so"));
    }
    out
}

unsafe fn load_sym<T>(handle: *mut c_void, name: &CStr) -> Option<T> {
    let sym = libc::dlsym(handle, name.as_ptr());
    if sym.is_null() {
        None
    } else {
        Some(std::mem::transmute_copy(&sym))
    }
}

unsafe fn open_real() -> Option<RealLib> {
    for path in candidate_real_paths() {
        if !path.is_file() {
            continue;
        }
        let cpath = CString::new(path.as_os_str().as_bytes()).ok()?;
        let handle = libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if handle.is_null() {
            continue;
        }
        let api = RealApi {
            register_callback: load_sym(handle, c"MsprofRegisterCallback")?,
            register_profile_callback: load_sym(handle, c"MsprofRegisterProfileCallback")?,
            reg_reporter_callback: load_sym(handle, c"profRegReporterCallback")?,
            reg_ctrl_callback: load_sym(handle, c"profRegCtrlCallback")?,
            reg_device_callback: load_sym(handle, c"profRegDeviceStateCallback")?,
            set_prof_command: load_sym(handle, c"profSetProfCommand")?,
            get_model_device: load_sym(handle, c"profGetDeviceIdByGeModelIdx")?,
            set_step_info: load_sym(handle, c"profSetStepInfo")?,
            reg_type_info: load_sym(handle, c"MsprofRegTypeInfo")?,
            init: load_sym(handle, c"MsprofInit")?,
            finalize: load_sym(handle, c"MsprofFinalize")?,
            report_api: load_sym(handle, c"MsprofReportApi")?,
            report_event: load_sym(handle, c"MsprofReportEvent")?,
            report_compact: load_sym(handle, c"MsprofReportCompactInfo")?,
            report_additional: load_sym(handle, c"MsprofReportAdditionalInfo")?,
            report_data: load_sym(handle, c"MsprofReportData")?,
            notify_set_device: load_sym(handle, c"MsprofNotifySetDevice")?,
            set_config: load_sym(handle, c"MsprofSetConfig")?,
            set_model_device: load_sym(handle, c"MsprofSetDeviceIdByGeModelIdx")?,
            unset_model_device: load_sym(handle, c"MsprofUnsetDeviceIdByGeModelIdx")?,
            start: load_sym(handle, c"MsprofStart")?,
            stop: load_sym(handle, c"MsprofStop")?,
            get_hash_id: load_sym(handle, c"MsprofGetHashId")?,
            str_to_id: load_sym(handle, c"MsprofStr2Id")?,
            sys_cycle_time: load_sym(handle, c"MsprofSysCycleTime")?,
        };
        log_once(&format!("forwarding to {}", path.display()));
        return Some(RealLib { handle, api });
    }
    None
}

fn real_lib() -> Option<parking_lot::MutexGuard<'static, Option<RealLib>>> {
    let mut guard = INIT.lock();
    if guard.is_none() && !INIT_FAILED.load(Ordering::Relaxed) {
        unsafe {
            *guard = open_real();
        }
        if guard.is_none() {
            INIT_FAILED.store(true, Ordering::Relaxed);
            if LOGGED_INIT
                .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                crate::log::warn(format!(
                    "real libprofapi not found; MSProf forward disabled. \
                     Set {ENV_REAL} or place {REAL_BASENAME} next to the shim."
                ));
            }
        }
    }
    Some(guard)
}

fn real_fn<T: Copy>(select: impl FnOnce(&RealApi) -> T) -> Option<T> {
    let guard = real_lib()?;
    guard.as_ref().map(|real| select(&real.api))
}

unsafe extern "C" fn stub_register(_: u32, _: ProfCommandHandle) -> i32 {
    0
}
unsafe extern "C" fn stub_register_profile(_: i32, _: *mut c_void, _: u32) -> i32 {
    0
}
unsafe extern "C" fn stub_reg_type(_: u16, _: u32, _: *const c_char) -> i32 {
    0
}
unsafe extern "C" fn stub_report_api(_: u32, _: *const c_void) -> i32 {
    0
}
unsafe extern "C" fn stub_init(_: u32, _: *mut c_void, _: u32) -> i32 {
    0
}
unsafe extern "C" fn stub_finalize() -> i32 {
    0
}
unsafe extern "C" fn stub_report_data(_: u32, _: u32, _: *mut c_void, _: u32) -> i32 {
    0
}
unsafe extern "C" fn stub_notify_set_device(_: u32, _: u32, _: bool) -> i32 {
    0
}
unsafe extern "C" fn stub_set_config(_: u32, _: *const c_char, _: usize) -> i32 {
    0
}
unsafe extern "C" fn stub_set_model_device(_: u32, _: u32) -> i32 {
    0
}
unsafe extern "C" fn stub_start_stop(_: u32, _: *const c_void, _: u32) -> i32 {
    0
}
unsafe extern "C" fn stub_report_blob(_: u32, _: *const c_void, _: u32) -> i32 {
    0
}
unsafe extern "C" fn stub_hash(_: *const c_char, _: u32) -> u64 {
    0
}
unsafe extern "C" fn stub_str_to_id(_: *const c_char, _: usize) -> u64 {
    0
}
unsafe extern "C" fn stub_time() -> u64 {
    0
}

pub fn forward_register(module_id: u32, handle: ProfCommandHandle) -> i32 {
    if let Some(function) = real_fn(|api| api.register_callback) {
        return unsafe { function(module_id, handle) };
    }
    unsafe { stub_register(module_id, handle) }
}

pub fn forward_register_profile(callback_type: i32, callback: *mut c_void, len: u32) -> i32 {
    if let Some(function) = real_fn(|api| api.register_profile_callback) {
        return unsafe { function(callback_type, callback, len) };
    }
    unsafe { stub_register_profile(callback_type, callback, len) }
}

pub fn forward_legacy_register(callback: *mut c_void, kind: u8) -> i32 {
    if let Some(function) = real_fn(|api| match kind {
        0 => api.reg_reporter_callback,
        1 => api.reg_ctrl_callback,
        _ => api.reg_device_callback,
    }) {
        return unsafe { function(callback) };
    }
    0
}

pub fn forward_set_prof_command(command: *mut c_void, len: u32) -> i32 {
    if let Some(function) = real_fn(|api| api.set_prof_command) {
        return unsafe { function(command, len) };
    }
    0
}

pub fn forward_get_model_device(model: u32, device: *mut u32) -> i32 {
    if let Some(function) = real_fn(|api| api.get_model_device) {
        return unsafe { function(model, device) };
    }
    0
}

pub fn forward_set_step_info(index: u64, tag: u16, stream: *mut c_void) -> i32 {
    if let Some(function) = real_fn(|api| api.set_step_info) {
        return unsafe { function(index, tag, stream) };
    }
    0
}

pub fn forward_reg_type_info(level: u16, type_id: u32, type_name: *const c_char) -> i32 {
    if let Some(function) = real_fn(|api| api.reg_type_info) {
        return unsafe { function(level, type_id, type_name) };
    }
    unsafe { stub_reg_type(level, type_id, type_name) }
}

pub fn forward_report_api(aging: u32, api: *const c_void) -> i32 {
    if let Some(function) = real_fn(|real| real.report_api) {
        return unsafe { function(aging, api) };
    }
    unsafe { stub_report_api(aging, api) }
}

pub fn forward_init(data_type: u32, data: *mut c_void, len: u32) -> i32 {
    if let Some(function) = real_fn(|api| api.init) {
        return unsafe { function(data_type, data, len) };
    }
    unsafe { stub_init(data_type, data, len) }
}

pub fn forward_finalize() -> i32 {
    if let Some(function) = real_fn(|api| api.finalize) {
        return unsafe { function() };
    }
    unsafe { stub_finalize() }
}

pub fn forward_report_event(aging: u32, event: *const c_void) -> i32 {
    if let Some(function) = real_fn(|api| api.report_event) {
        return unsafe { function(aging, event) };
    }
    unsafe { stub_report_api(aging, event) }
}

pub fn forward_report_data(module_id: u32, kind: u32, data: *mut c_void, len: u32) -> i32 {
    if let Some(function) = real_fn(|api| api.report_data) {
        return unsafe { function(module_id, kind, data, len) };
    }
    unsafe { stub_report_data(module_id, kind, data, len) }
}

pub fn forward_notify_set_device(chip_id: u32, device_id: u32, is_open: bool) -> i32 {
    if let Some(function) = real_fn(|api| api.notify_set_device) {
        return unsafe { function(chip_id, device_id, is_open) };
    }
    unsafe { stub_notify_set_device(chip_id, device_id, is_open) }
}

pub fn forward_set_config(kind: u32, config: *const c_char, len: usize) -> i32 {
    if let Some(function) = real_fn(|api| api.set_config) {
        return unsafe { function(kind, config, len) };
    }
    unsafe { stub_set_config(kind, config, len) }
}

pub fn forward_set_model_device(model: u32, device: u32, unset: bool) -> i32 {
    if let Some(function) = real_fn(|api| {
        if unset {
            api.unset_model_device
        } else {
            api.set_model_device
        }
    }) {
        return unsafe { function(model, device) };
    }
    unsafe { stub_set_model_device(model, device) }
}

pub fn forward_start_stop(kind: u32, data: *const c_void, len: u32, stop: bool) -> i32 {
    if let Some(function) = real_fn(|api| if stop { api.stop } else { api.start }) {
        return unsafe { function(kind, data, len) };
    }
    unsafe { stub_start_stop(kind, data, len) }
}

pub fn forward_report_compact(aging: u32, data: *const c_void, len: u32) -> i32 {
    if let Some(function) = real_fn(|api| api.report_compact) {
        return unsafe { function(aging, data, len) };
    }
    unsafe { stub_report_blob(aging, data, len) }
}

pub fn forward_report_additional(aging: u32, data: *const c_void, len: u32) -> i32 {
    if let Some(function) = real_fn(|api| api.report_additional) {
        return unsafe { function(aging, data, len) };
    }
    unsafe { stub_report_blob(aging, data, len) }
}

pub fn forward_get_hash_id(hash_info: *const c_char, length: u32) -> u64 {
    if hash_info.is_null() || length == 0 {
        return 0;
    }
    if let Some(function) = real_fn(|api| api.get_hash_id) {
        return unsafe { function(hash_info, length) };
    }
    unsafe { stub_hash(hash_info, length) }
}

pub fn forward_str_to_id(hash_info: *const c_char, length: usize) -> u64 {
    if let Some(function) = real_fn(|api| api.str_to_id) {
        return unsafe { function(hash_info, length) };
    }
    unsafe { stub_str_to_id(hash_info, length) }
}

pub fn forward_sys_cycle_time() -> u64 {
    if let Some(function) = real_fn(|api| api.sys_cycle_time) {
        return unsafe { function() };
    }
    unsafe { stub_time() }
}
