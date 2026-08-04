//! Best-effort layouts for CANN MSProf structs (toolchain/prof_api.h).
//!
//! Field order follows open HCCL usage in task_profiling.cc / profiling_manager.cc.
//! Validate with `sizeof` checks at runtime; CANN version drift may require updates.

use std::mem::size_of;

const MSPROF_REPORT_DATA_MAGIC: u16 = 0x5a5a;

/// HCCL passes `sizeof(MsprofHcclInfo)` bytes in AdditionalInfo.data for task reports.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MsprofHcclInfo {
    pub item_id: u64,
    pub ccl_tag: u64,
    pub group_name: u64,
    pub local_rank: u32,
    pub remote_rank: u32,
    pub rank_size: u32,
    pub workflow_mode: u32,
    pub plane_id: u32,
    pub ctx_id: u32,
    pub notify_id: u64,
    pub stage: u32,
    pub role: u32,
    pub duration_estimated: f64,
    pub src_addr: u64,
    pub dst_addr: u64,
    pub data_size: u64,
    pub op_type: u32,
    pub data_type: u32,
    pub link_type: u32,
    pub transport_type: u32,
    pub rdma_type: u32,
    pub reserve2: u32,
}

pub const MSPROF_HCCL_INFO_MIN: usize = size_of::<MsprofHcclInfo>();

/// Observed HCCL MsprofApi initialization pattern.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MsprofApi {
    pub magic_number: u16,
    pub level: u16,
    pub type_id: u32,
    pub thread_id: u32,
    pub reserve2: u32,
    pub begin_time: u64,
    pub end_time: u64,
    pub item_id: u64,
}

pub const MSPROF_API_SIZE: usize = size_of::<MsprofApi>();

/// Header shared by MsprofAdditionalInfo and MsprofCompactInfo.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MsprofBlobHeader {
    pub magic_number: u16,
    pub level: u16,
    pub type_id: u32,
    pub thread_id: u32,
    pub data_len: u32,
    pub time_stamp: u64,
}

pub type MsprofAdditionalInfoHeader = MsprofBlobHeader;
pub type MsprofCompactInfoHeader = MsprofBlobHeader;

pub const MSPROF_BLOB_HEADER: usize = size_of::<MsprofBlobHeader>();
pub const MSPROF_ADDITIONAL_HEADER: usize = MSPROF_BLOB_HEADER;

/// `CallMsprofReportHostHcclOpInfo` payload inside MsprofCompactInfo.data.
#[derive(Clone, Copy, Default)]
pub struct MsprofHCCLOPInfo {
    pub relay: u8,
    pub retry: u8,
    pub data_type: u8,
    pub alg_type: u64,
    pub count: u64,
    pub group_name: u64,
}

// CANN 8.5 declares this payload under `#pragma pack(1)`: one bitfield
// storage byte, one data-type byte, then three unaligned u64 fields.
pub const MSPROF_HCCL_OP_INFO_MIN: usize = 26;

/// `CallMsprofReportContextIdInfo` payload.
#[derive(Clone, Copy, Default)]
pub struct MsprofContextIdInfo {
    pub _op_name: u64,
    pub ctx_id_num: u32,
    pub ctx_ids: [u32; 2],
}

pub const MSPROF_CONTEXT_ID_INFO_MIN: usize = 20;

/// Prefix of `ProfilingDeviceCommResInfo` from hccl_communicator_host.cc.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ProfilingDeviceCommResInfoHeader {
    pub group_name: u64,
    pub rank_size: u32,
    pub rank_id: u32,
    pub usr_rank_id: u32,
    pub aicpu_kfc_stream_id: u32,
}

pub const MSPROF_MC2_INFO_SIZE: usize = 64;

/// ProfTaskType::TASK_HCCL_INFO
pub const MSPROF_REPORT_NODE_LEVEL: u16 = 10_000;
pub const MSPROF_REPORT_HCCL_LEVEL: u16 = 5_500;
pub const MSPROF_REPORT_AICPU_LEVEL: u16 = 6_000;
pub const MSPROF_REPORT_ACL_LEVEL: u16 = 20_000;
pub const MSPROF_REPORT_NODE_LAUNCH_TYPE: u32 = 5;
pub const MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE: u32 = 0x07_0000;
pub const MSPROF_REPORT_API_BASE_MASK: u32 = 0xFF_0000;
pub const MSPROF_REPORT_HCCL_MASTER_TYPE: u32 = 0x01_0001;
pub const MSPROF_REPORT_HCCL_SLAVE_TYPE: u32 = 0x01_0002;
pub const MSPROF_REPORT_NODE_CONTEXT_ID_TYPE: u32 = 4;
pub const MSPROF_REPORT_NODE_HCCL_OP_TYPE: u32 = 10;
pub const MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE: u32 = 12;
pub const MSPROF_REPORT_AICPU_MC2_HCCL_INFO_TYPE: u32 = 6;

pub fn decode_plane(plane_id: u32) -> (i32, i32, i32) {
    let id = plane_id as u64;
    let plane_index = ((id >> 28) & 0xF) as i32;
    let rank_size_plane = ((id >> 16) & 0xFFF) as i32;
    let rank_in_plane = (id & 0xFFFF) as i32;
    (plane_index, rank_in_plane, rank_size_plane)
}

pub fn read_api(ptr: *const u8, len: u32) -> Option<MsprofApi> {
    if ptr.is_null() || (len as usize) < MSPROF_API_SIZE {
        return None;
    }
    let value = unsafe { std::ptr::read_unaligned(ptr as *const MsprofApi) };
    (value.magic_number == MSPROF_REPORT_DATA_MAGIC).then_some(value)
}

pub fn read_blob_header(ptr: *const u8, len: u32) -> Option<MsprofBlobHeader> {
    if ptr.is_null() || (len as usize) < MSPROF_BLOB_HEADER {
        return None;
    }
    let value = unsafe { std::ptr::read_unaligned(ptr as *const MsprofBlobHeader) };
    (value.magic_number == MSPROF_REPORT_DATA_MAGIC).then_some(value)
}

pub fn read_additional_header(ptr: *const u8, len: u32) -> Option<MsprofAdditionalInfoHeader> {
    read_blob_header(ptr, len)
}

pub fn read_compact_header(ptr: *const u8, len: u32) -> Option<MsprofCompactInfoHeader> {
    read_blob_header(ptr, len)
}

pub fn read_hccl_info(data: *const u8, data_len: u32) -> Option<MsprofHcclInfo> {
    if data.is_null() || (data_len as usize) < MSPROF_HCCL_INFO_MIN {
        return None;
    }
    Some(unsafe { std::ptr::read_unaligned(data as *const MsprofHcclInfo) })
}

pub fn read_hccl_op_info(data: *const u8, data_len: u32) -> Option<MsprofHCCLOPInfo> {
    if data.is_null() || (data_len as usize) < MSPROF_HCCL_OP_INFO_MIN {
        return None;
    }
    let flags = unsafe { *data };
    let data_type = unsafe { *data.add(1) };
    let alg_type = unsafe { std::ptr::read_unaligned(data.add(2) as *const u64) };
    let count = unsafe { std::ptr::read_unaligned(data.add(10) as *const u64) };
    let group_name = unsafe { std::ptr::read_unaligned(data.add(18) as *const u64) };
    Some(MsprofHCCLOPInfo {
        relay: flags & 1,
        retry: (flags >> 1) & 1,
        data_type,
        alg_type,
        count,
        group_name,
    })
}

pub fn read_context_id_info(data: *const u8, data_len: u32) -> Option<MsprofContextIdInfo> {
    if data.is_null() || (data_len as usize) < MSPROF_CONTEXT_ID_INFO_MIN {
        return None;
    }
    Some(MsprofContextIdInfo {
        _op_name: unsafe { std::ptr::read_unaligned(data as *const u64) },
        ctx_id_num: unsafe { std::ptr::read_unaligned(data.add(8) as *const u32) },
        ctx_ids: [
            unsafe { std::ptr::read_unaligned(data.add(12) as *const u32) },
            unsafe { std::ptr::read_unaligned(data.add(16) as *const u32) },
        ],
    })
}

pub struct Mc2CommInfo {
    pub header: ProfilingDeviceCommResInfoHeader,
    pub comm_stream_size: u32,
    pub comm_stream_ids: Vec<u32>,
}

pub fn read_mc2_comm_info(data: *const u8, data_len: u32) -> Option<Mc2CommInfo> {
    if data.is_null() || (data_len as usize) < MSPROF_MC2_INFO_SIZE {
        return None;
    }
    let header = ProfilingDeviceCommResInfoHeader {
        group_name: unsafe { std::ptr::read_unaligned(data as *const u64) },
        rank_size: unsafe { std::ptr::read_unaligned(data.add(8) as *const u32) },
        rank_id: unsafe { std::ptr::read_unaligned(data.add(12) as *const u32) },
        usr_rank_id: unsafe { std::ptr::read_unaligned(data.add(16) as *const u32) },
        aicpu_kfc_stream_id: unsafe { std::ptr::read_unaligned(data.add(20) as *const u32) },
    };
    let comm_stream_size = unsafe { std::ptr::read_unaligned(data.add(24) as *const u32) }.min(8);
    let n = comm_stream_size as usize;
    let mut comm_stream_ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = unsafe { std::ptr::read_unaligned(data.add(28 + i * 4) as *const u32) };
        comm_stream_ids.push(id);
    }

    Some(Mc2CommInfo {
        header,
        comm_stream_size,
        comm_stream_ids,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdditionalKind {
    HcclTask,
    Mc2Comm,
    ContextId,
    Unknown,
}

pub fn classify_additional(
    level: u16,
    type_id: u32,
    type_name: &str,
    data_len: u32,
) -> AdditionalKind {
    if type_name.contains("mc2_comm_info")
        || (level == MSPROF_REPORT_NODE_LEVEL && type_id == MSPROF_REPORT_NODE_MC2_COMMINFO_TYPE)
    {
        return AdditionalKind::Mc2Comm;
    }
    if type_name.contains("context_id_info")
        || (level == MSPROF_REPORT_NODE_LEVEL && type_id == MSPROF_REPORT_NODE_CONTEXT_ID_TYPE)
    {
        return AdditionalKind::ContextId;
    }
    if type_name.contains("hccl_info")
        || level == MSPROF_REPORT_HCCL_LEVEL
        || (level == MSPROF_REPORT_AICPU_LEVEL && type_id == MSPROF_REPORT_AICPU_MC2_HCCL_INFO_TYPE)
        || data_len as usize == MSPROF_HCCL_INFO_MIN
    {
        return AdditionalKind::HcclTask;
    }
    if data_len as usize == MSPROF_MC2_INFO_SIZE {
        return AdditionalKind::Mc2Comm;
    }
    AdditionalKind::Unknown
}

pub fn is_hccl_op_compact(level: u16, type_id: u32, type_name: &str, data_len: u32) -> bool {
    type_name.contains("hccl_op")
        || ((level == MSPROF_REPORT_NODE_LEVEL || level == MSPROF_REPORT_AICPU_LEVEL)
            && type_id == MSPROF_REPORT_NODE_HCCL_OP_TYPE)
        || data_len as usize == MSPROF_HCCL_OP_INFO_MIN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hccl_info_size_sane() {
        assert_eq!(MSPROF_HCCL_INFO_MIN, 120);
        assert_eq!(MSPROF_API_SIZE, 40);
        assert_eq!(MSPROF_BLOB_HEADER, 24);
        assert_eq!(MSPROF_HCCL_OP_INFO_MIN, 26);
        assert_eq!(MSPROF_MC2_INFO_SIZE, 64);
    }

    #[test]
    fn decode_plane_bits() {
        // plane=3, rank_size=8, rank=5 -> (3<<28)|(8<<16)|5
        let plane_id = (3u64 << 28) | (8u64 << 16) | 5;
        assert_eq!(decode_plane(plane_id as u32), (3, 5, 8));
    }

    #[test]
    fn classify_task_by_len() {
        assert_eq!(
            classify_additional(0, 99, "", MSPROF_HCCL_INFO_MIN as u32),
            AdditionalKind::HcclTask
        );
    }

    #[test]
    fn classifies_cann_85_level_and_type_ids() {
        assert_eq!(
            classify_additional(MSPROF_REPORT_NODE_LEVEL, 4, "", 232),
            AdditionalKind::ContextId
        );
        assert_eq!(
            classify_additional(MSPROF_REPORT_NODE_LEVEL, 12, "", 64),
            AdditionalKind::Mc2Comm
        );
        assert!(is_hccl_op_compact(MSPROF_REPORT_NODE_LEVEL, 10, "", 26));
    }

    #[test]
    fn parses_cann_85_compact_hccl_op_bytes() {
        let bytes = [
            0x02, 0x04, 0x94, 0x48, 0x77, 0xc5, 0x96, 0xf8, 0xea, 0xc6, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x0c, 0x9f, 0x5f, 0x0f, 0x0c, 0xfd, 0x5a, 0x51,
        ];
        let op = read_hccl_op_info(bytes.as_ptr(), bytes.len() as u32).unwrap();
        assert_eq!(op.relay, 0);
        assert_eq!(op.retry, 1);
        assert_eq!(op.data_type, 4);
        assert_eq!(op.count, 1);
        assert_eq!(op.group_name, 0x515afd0c0f5f9f0c);
    }

    #[test]
    fn parses_cann_85_mc2_bytes() {
        let mut bytes = [0u8; MSPROF_MC2_INFO_SIZE];
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&197u32.to_le_bytes());
        bytes[24..28].copy_from_slice(&8u32.to_le_bytes());
        for (index, stream_id) in (233u32..=240).enumerate() {
            let offset = 28 + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&stream_id.to_le_bytes());
        }
        let mc2 = read_mc2_comm_info(bytes.as_ptr(), bytes.len() as u32).unwrap();
        assert_eq!(mc2.header.rank_size, 2);
        assert_eq!(mc2.header.aicpu_kfc_stream_id, 197);
        assert_eq!(mc2.comm_stream_size, 8);
        assert_eq!(mc2.comm_stream_ids, (233u32..=240).collect::<Vec<_>>());
    }
}
