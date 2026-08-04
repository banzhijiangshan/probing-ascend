//! Resolve MSProf `item_id` hashes to human-readable names.

use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, Ordering};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::msprof::{
    MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE, MSPROF_REPORT_ACL_LEVEL, MSPROF_REPORT_API_BASE_MASK,
    MSPROF_REPORT_HCCL_LEVEL, MSPROF_REPORT_HCCL_MASTER_TYPE, MSPROF_REPORT_HCCL_SLAVE_TYPE,
    MSPROF_REPORT_NODE_LAUNCH_TYPE, MSPROF_REPORT_NODE_LEVEL,
};

static REGISTRY: Lazy<Mutex<NameRegistry>> = Lazy::new(|| Mutex::new(NameRegistry::new()));
static PRESEEDED: AtomicBool = AtomicBool::new(false);

/// ProfTaskType names from HCCL `PROF_TASK_OP_NAME`.
static KNOWN_TASK_NAMES: &[&str] = &[
    "hccl_info",
    "Memcpy",
    "RDMASend",
    "Reduce_Inline",
    "Reduce_TBE",
    "Notify_Record",
    "Notify_Wait",
    "StageX_StepX",
    "Flag",
    "End",
    "Multi_Thread",
    "Launch_Ffts",
    "AivKernel",
    "Wait_Some",
    "Coll_Recv_Lookup_Request",
    "Coll_Recv_Update_Request",
    "Isend_Update_Response",
    "Isend_Lookup_Response",
    "Update_Imrecv",
    "Update_Global_Reduce",
    "Lookup_Response_Memcpy",
    "Lookup_Response_Isend",
    "Share_Memory_Isend_Record",
    "Abort_Self",
    "Service_Cancel",
    "Destroy_Resource",
    "Event_Wait",
    "unknown",
];

/// HcclCMDType names from HCCL `PROF_OP_NAME`.
static KNOWN_OP_NAMES: &[&str] = &[
    "hcom_invalid_",
    "hcom_broadcast_",
    "hcom_allReduce_",
    "hcom_reduce_",
    "hcom_send_",
    "hcom_receive_",
    "hcom_allGather_",
    "hcom_reduceScatter_",
    "hcom_scatter_",
    "hcom_alltoall_",
    "hcom_alltoallv_",
    "hcom_allGatherv_",
    "hcom_reduceScatterv_",
    "hcom_alltoallvc_",
    "hcom_batchSendRecv_",
    "hccl_batchPut_",
    "hccl_batchGet_",
];

struct NameRegistry {
    hash_to_name: HashMap<u64, String>,
    type_id_labels: HashMap<u32, String>,
}

impl NameRegistry {
    fn new() -> Self {
        Self {
            hash_to_name: HashMap::new(),
            type_id_labels: HashMap::new(),
        }
    }

    fn insert_hash(&mut self, hash: u64, name: impl Into<String>) {
        if hash != 0 {
            self.hash_to_name.entry(hash).or_insert(name.into());
        }
    }

    fn lookup(&self, hash: u64) -> String {
        self.hash_to_name.get(&hash).cloned().unwrap_or_default()
    }

    fn label_for_type(&self, type_id: u32) -> String {
        self.type_id_labels
            .get(&type_id)
            .cloned()
            .unwrap_or_default()
    }
}

pub fn register_type_info(
    type_id: u32,
    type_name: *const c_char,
    hash_fn: impl Fn(*const c_char, u32) -> u64,
) {
    if type_name.is_null() {
        return;
    }
    let Ok(name) = unsafe { CStr::from_ptr(type_name) }.to_str() else {
        return;
    };
    let mut reg = REGISTRY.lock();
    reg.type_id_labels
        .entry(type_id)
        .or_insert_with(|| name.to_string());
    let hash = hash_fn(type_name, name.len() as u32);
    reg.insert_hash(hash, name);
}

pub fn register_hash_string(hash_info: *const c_char, length: u32, hash: u64) {
    if hash_info.is_null() || length == 0 || hash == 0 {
        return;
    }
    let bytes = unsafe { std::slice::from_raw_parts(hash_info as *const u8, length as usize) };
    let Ok(name) = std::str::from_utf8(bytes) else {
        return;
    };
    REGISTRY.lock().insert_hash(hash, name);
}

pub fn preseed_hashes(hash_fn: impl Fn(*const c_char, u32) -> u64) {
    if PRESEEDED.swap(true, Ordering::Relaxed) {
        return;
    }
    let mut reg = REGISTRY.lock();
    for name in KNOWN_TASK_NAMES.iter().chain(KNOWN_OP_NAMES.iter()) {
        let c = std::ffi::CString::new(*name).expect("static name");
        let hash = hash_fn(c.as_ptr(), name.len() as u32);
        reg.insert_hash(hash, *name);
    }
}

pub fn lookup_hash(hash: u64) -> String {
    REGISTRY.lock().lookup(hash)
}

pub fn lookup_type_id(type_id: u32) -> String {
    REGISTRY.lock().label_for_type(type_id)
}

/// Classify `MsprofReportApi` rows using level/type and resolved item name.
pub fn classify_api_event(level: u16, type_id: u32, item_id: u64) -> &'static str {
    let name = lookup_hash(item_id);
    if name.starts_with("hcom_") || name.starts_with("hccl_") {
        return match (level, type_id) {
            (MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_LAUNCH_TYPE) => "node_launch",
            (MSPROF_REPORT_HCCL_LEVEL, MSPROF_REPORT_HCCL_MASTER_TYPE) => "task_master",
            (MSPROF_REPORT_HCCL_LEVEL, MSPROF_REPORT_HCCL_SLAVE_TYPE) => "task_slave",
            (MSPROF_REPORT_ACL_LEVEL, ty)
                if ty & MSPROF_REPORT_API_BASE_MASK == MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE =>
            {
                "host_hccl_op"
            }
            // Node-level rows describe launch/execution metadata, not a second
            // invocation of the user-visible collective.
            (MSPROF_REPORT_NODE_LEVEL, _) => "hccl_node",
            (MSPROF_REPORT_ACL_LEVEL, _) => "host_acl",
            // Compatibility fallback for older CANN releases whose compact
            // level/type constants differ but still register canonical names.
            _ => "host_hccl_op",
        };
    }
    if KNOWN_TASK_NAMES.contains(&name.as_str()) || name.ends_with("Kernel") {
        return match type_id {
            2 => "task_slave",
            _ => "task_master",
        };
    }
    match (level, type_id) {
        (MSPROF_REPORT_ACL_LEVEL, _) => "host_acl",
        // A node-level record is not necessarily HCCL.  Only the name-aware
        // branches above may classify generic node records as HCCL events.
        (MSPROF_REPORT_NODE_LEVEL, _) => "api_other",
        (MSPROF_REPORT_HCCL_LEVEL, MSPROF_REPORT_HCCL_MASTER_TYPE) => "task_master",
        (MSPROF_REPORT_HCCL_LEVEL, MSPROF_REPORT_HCCL_SLAVE_TYPE) => "task_slave",
        _ => "api_other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_by_name_prefix() {
        let mut reg = REGISTRY.lock();
        reg.insert_hash(42, "hcom_allReduce_");
        drop(reg);
        assert_eq!(
            classify_api_event(MSPROF_REPORT_ACL_LEVEL, 0x07_0002, 42),
            "host_hccl_op"
        );
        assert_eq!(
            classify_api_event(MSPROF_REPORT_NODE_LEVEL, MSPROF_REPORT_NODE_LAUNCH_TYPE, 42),
            "node_launch"
        );
        assert_eq!(
            classify_api_event(MSPROF_REPORT_HCCL_LEVEL, MSPROF_REPORT_HCCL_MASTER_TYPE, 42),
            "task_master"
        );
    }
}
