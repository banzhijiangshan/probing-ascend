//! `libprofapi.so` shim — intercept MSProf, write `hccl.*` memtables, forward to CANN.

#![allow(clippy::missing_safety_doc)]
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

#[cfg(target_os = "linux")]
mod forward;
mod log;
mod msprof;
mod names;
pub mod tables;
pub use tables::register_docs;
mod writer;

#[cfg(not(target_os = "linux"))]
mod forward {
    use std::os::raw::{c_char, c_void};
    type ProfCommandHandle = Option<unsafe extern "C" fn(u32, *mut c_void, u32) -> i32>;
    pub fn forward_register(_: u32, _: ProfCommandHandle) -> i32 {
        0
    }
    pub fn forward_register_profile(_: i32, _: *mut c_void, _: u32) -> i32 {
        0
    }
    pub fn forward_legacy_register(_: *mut c_void, _: u8) -> i32 {
        0
    }
    pub fn forward_set_prof_command(_: *mut c_void, _: u32) -> i32 {
        0
    }
    pub fn forward_get_model_device(_: u32, _: *mut u32) -> i32 {
        0
    }
    pub fn forward_set_step_info(_: u64, _: u16, _: *mut c_void) -> i32 {
        0
    }
    pub fn forward_reg_type_info(_: u16, _: u32, _: *const c_char) -> i32 {
        0
    }
    pub fn forward_report_api(_: u32, _: *const c_void) -> i32 {
        0
    }
    pub fn forward_init(_: u32, _: *mut c_void, _: u32) -> i32 {
        0
    }
    pub fn forward_finalize() -> i32 {
        0
    }
    pub fn forward_report_event(_: u32, _: *const c_void) -> i32 {
        0
    }
    pub fn forward_report_data(_: u32, _: u32, _: *mut c_void, _: u32) -> i32 {
        0
    }
    pub fn forward_notify_set_device(_: u32, _: u32, _: bool) -> i32 {
        0
    }
    pub fn forward_set_config(_: u32, _: *const c_char, _: usize) -> i32 {
        0
    }
    pub fn forward_set_model_device(_: u32, _: u32, _: bool) -> i32 {
        0
    }
    pub fn forward_start_stop(_: u32, _: *const c_void, _: u32, _: bool) -> i32 {
        0
    }
    pub fn forward_report_compact(_: u32, _: *const c_void, _: u32) -> i32 {
        0
    }
    pub fn forward_report_additional(_: u32, _: *const c_void, _: u32) -> i32 {
        0
    }
    pub fn forward_get_hash_id(_: *const c_char, _: u32) -> u64 {
        0
    }
    pub fn forward_str_to_id(_: *const c_char, _: usize) -> u64 {
        0
    }
    pub fn forward_sys_cycle_time() -> u64 {
        0
    }
}

pub use tables::{
    collectives_schema, context_ids_schema, host_ops_schema, mc2_streams_schema, tasks_schema,
    COLLECTIVES_FILE, CONTEXT_IDS_FILE, HOST_OPS_FILE, MC2_STREAMS_FILE, TASKS_FILE,
};

use std::os::raw::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::msprof::{
    classify_additional, is_hccl_op_compact, read_additional_header, read_api, read_compact_header,
    read_context_id_info, read_hccl_info, read_hccl_op_info, read_mc2_comm_info, AdditionalKind,
    MSPROF_ADDITIONAL_HEADER, MSPROF_BLOB_HEADER,
};
use crate::names::{lookup_type_id, preseed_hashes};
use crate::writer::HcclWriter;

static WRITER: Lazy<Mutex<HcclWriter>> = Lazy::new(|| Mutex::new(HcclWriter::new()));
static BLOB_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);

type ProfCommandHandle = Option<unsafe extern "C" fn(u32, *mut c_void, u32) -> i32>;

fn hash_fn(s: *const std::os::raw::c_char, l: u32) -> u64 {
    forward::forward_get_hash_id(s, l)
}

fn ensure_names() {
    preseed_hashes(hash_fn);
}

fn log_blob(kind: &str, ptr: *const c_void, len: u32) {
    if std::env::var_os("PROBING_HCCL_SHIM_LOG").is_none() || ptr.is_null() || len < 24 {
        return;
    }
    let raw = ptr as *const u8;
    let level = unsafe { std::ptr::read_unaligned(raw.add(2) as *const u16) };
    let type_id = unsafe { std::ptr::read_unaligned(raw.add(4) as *const u32) };
    let interesting =
        level == 5_500 || level == 6_000 || (level == 10_000 && matches!(type_id, 4 | 10 | 12));
    if !interesting || BLOB_LOG_COUNT.fetch_add(1, Ordering::Relaxed) >= 32 {
        return;
    }
    let prefix_len = (len as usize).min(64);
    let prefix = unsafe { std::slice::from_raw_parts(raw, prefix_len) }.to_vec();
    crate::log::info(format!(
        "{kind} len={len} level={level} type={type_id} prefix={prefix:02x?}"
    ));
}

fn capture_api(aging: u32, ptr: *const c_void) {
    if ptr.is_null() {
        return;
    }
    ensure_names();
    if let Some(api) = read_api(ptr as *const u8, crate::msprof::MSPROF_API_SIZE as u32) {
        WRITER.lock().record_api(aging, &api);
    }
}

fn capture_compact(_aging: u32, ptr: *const c_void, len: u32) {
    if ptr.is_null() {
        return;
    }
    log_blob("compact", ptr, len);
    ensure_names();
    let Some(header) = read_compact_header(ptr as *const u8, len) else {
        return;
    };
    let data_ptr = unsafe { (ptr as *const u8).add(MSPROF_BLOB_HEADER) };
    let type_name = lookup_type_id(header.type_id);
    if is_hccl_op_compact(header.level, header.type_id, &type_name, header.data_len) {
        if let Some(op) = read_hccl_op_info(data_ptr, header.data_len) {
            WRITER.lock().record_compact_hccl_op(&header, &op);
        }
    }
}

fn capture_additional(_aging: u32, ptr: *const c_void, len: u32) {
    if ptr.is_null() {
        return;
    }
    log_blob("additional", ptr, len);
    ensure_names();
    let Some(header) = read_additional_header(ptr as *const u8, len) else {
        return;
    };
    let data_ptr = unsafe { (ptr as *const u8).add(MSPROF_ADDITIONAL_HEADER) };
    let type_name = lookup_type_id(header.type_id);
    match classify_additional(header.level, header.type_id, &type_name, header.data_len) {
        AdditionalKind::HcclTask => {
            if let Some(hccl) = read_hccl_info(data_ptr, header.data_len) {
                WRITER
                    .lock()
                    .record_task(&header, &hccl, header.data_len as i32);
            }
        }
        AdditionalKind::Mc2Comm => {
            if let Some(mc2) = read_mc2_comm_info(data_ptr, header.data_len) {
                WRITER.lock().record_mc2(&header, &mc2);
            }
        }
        AdditionalKind::ContextId => {
            if let Some(ctx) = read_context_id_info(data_ptr, header.data_len) {
                WRITER.lock().record_context(&header, &ctx);
            }
        }
        AdditionalKind::Unknown => {}
    }
}

#[cfg(target_os = "linux")]
mod export {
    use std::os::raw::{c_char, c_void};

    use super::*;

    #[no_mangle]
    pub unsafe extern "C" fn MsprofRegisterCallback(
        module_id: u32,
        handle: ProfCommandHandle,
    ) -> i32 {
        forward::forward_register(module_id, handle)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofRegisterProfileCallback(
        callback_type: i32,
        callback: *mut c_void,
        len: u32,
    ) -> i32 {
        let rc = forward::forward_register_profile(callback_type, callback, len);
        if std::env::var_os("PROBING_HCCL_SHIM_LOG").is_some() {
            crate::log::info(format!(
                "MsprofRegisterProfileCallback type={callback_type} len={len} rc={rc}"
            ));
        }
        rc
    }

    #[no_mangle]
    pub unsafe extern "C" fn profRegReporterCallback(callback: *mut c_void) -> i32 {
        forward::forward_legacy_register(callback, 0)
    }

    #[no_mangle]
    pub unsafe extern "C" fn profRegCtrlCallback(callback: *mut c_void) -> i32 {
        forward::forward_legacy_register(callback, 1)
    }

    #[no_mangle]
    pub unsafe extern "C" fn profRegDeviceStateCallback(callback: *mut c_void) -> i32 {
        forward::forward_legacy_register(callback, 2)
    }

    #[no_mangle]
    pub unsafe extern "C" fn profSetProfCommand(command: *mut c_void, len: u32) -> i32 {
        forward::forward_set_prof_command(command, len)
    }

    #[no_mangle]
    pub unsafe extern "C" fn profGetDeviceIdByGeModelIdx(model: u32, device: *mut u32) -> i32 {
        forward::forward_get_model_device(model, device)
    }

    #[no_mangle]
    pub unsafe extern "C" fn profSetStepInfo(index: u64, tag: u16, stream: *mut c_void) -> i32 {
        forward::forward_set_step_info(index, tag, stream)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofRegTypeInfo(
        level: u16,
        type_id: u32,
        type_name: *const c_char,
    ) -> i32 {
        ensure_names();
        crate::names::register_type_info(type_id, type_name, hash_fn);
        forward::forward_reg_type_info(level, type_id, type_name)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofReportApi(aging_flag: u32, api: *const c_void) -> i32 {
        capture_api(aging_flag, api);
        forward::forward_report_api(aging_flag, api)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofInit(data_type: u32, data: *mut c_void, len: u32) -> i32 {
        let rc = forward::forward_init(data_type, data, len);
        if std::env::var_os("PROBING_HCCL_SHIM_LOG").is_some() {
            crate::log::info(format!("MsprofInit type={data_type} len={len} rc={rc}"));
        }
        rc
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofFinalize() -> i32 {
        forward::forward_finalize()
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofReportEvent(aging_flag: u32, event: *const c_void) -> i32 {
        forward::forward_report_event(aging_flag, event)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofReportData(
        module_id: u32,
        kind: u32,
        data: *mut c_void,
        len: u32,
    ) -> i32 {
        forward::forward_report_data(module_id, kind, data, len)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofNotifySetDevice(
        chip_id: u32,
        device_id: u32,
        is_open: bool,
    ) -> i32 {
        forward::forward_notify_set_device(chip_id, device_id, is_open)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofSetConfig(kind: u32, config: *const c_char, len: usize) -> i32 {
        let rc = forward::forward_set_config(kind, config, len);
        if std::env::var_os("PROBING_HCCL_SHIM_LOG").is_some() {
            crate::log::info(format!("MsprofSetConfig kind={kind} len={len} rc={rc}"));
        }
        rc
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofSetDeviceIdByGeModelIdx(model: u32, device: u32) -> i32 {
        forward::forward_set_model_device(model, device, false)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofUnsetDeviceIdByGeModelIdx(model: u32, device: u32) -> i32 {
        forward::forward_set_model_device(model, device, true)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofStart(kind: u32, data: *const c_void, len: u32) -> i32 {
        forward::forward_start_stop(kind, data, len, false)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofStop(kind: u32, data: *const c_void, len: u32) -> i32 {
        forward::forward_start_stop(kind, data, len, true)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofReportCompactInfo(
        aging_flag: u32,
        data: *const c_void,
        length: u32,
    ) -> i32 {
        capture_compact(aging_flag, data, length);
        forward::forward_report_compact(aging_flag, data, length)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofReportAdditionalInfo(
        aging_flag: u32,
        data: *const c_void,
        length: u32,
    ) -> i32 {
        capture_additional(aging_flag, data, length);
        forward::forward_report_additional(aging_flag, data, length)
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofGetHashId(hash_info: *const c_char, length: u32) -> u64 {
        ensure_names();
        let hash = forward::forward_get_hash_id(hash_info, length);
        crate::names::register_hash_string(hash_info, length, hash);
        hash
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofStr2Id(hash_info: *const c_char, length: usize) -> u64 {
        let hash = forward::forward_str_to_id(hash_info, length);
        crate::names::register_hash_string(hash_info, length.min(u32::MAX as usize) as u32, hash);
        hash
    }

    #[no_mangle]
    pub unsafe extern "C" fn MsprofSysCycleTime() -> u64 {
        forward::forward_sys_cycle_time()
    }
}

#[cfg(not(target_os = "linux"))]
mod stub {
    pub const BUILD_NOTE: &str = "probing-hccl-shim: libprofapi.so built on Linux only";
}
