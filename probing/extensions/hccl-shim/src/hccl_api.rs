//! Direct HCCL API interposition for normal (non-MSProf) DDP training.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;
use std::time::Instant;

type HcclResult = c_int;
type HcclDataType = c_int;
type HcclReduceOp = c_int;
type HcclComm = *mut c_void;
type AclrtStream = *mut c_void;

unsafe fn resolve(name: &'static [u8]) -> *mut c_void {
    // Preserve the LD_PRELOAD chain first so another interposer (for example
    // the external fail-slow injector) remains observable inside this timer.
    let next = libc::dlsym(libc::RTLD_NEXT, name.as_ptr().cast());
    if !next.is_null() {
        return next;
    }

    // In CANN containers libhccl is commonly loaded after the preload shim,
    // so RTLD_NEXT alone can miss it during early process startup. Resolve
    // from an explicit handle as the fallback used for standalone probing.
    let handle = libc::dlopen(
        b"libhccl.so\0".as_ptr().cast(),
        libc::RTLD_LAZY | libc::RTLD_GLOBAL,
    );
    if !handle.is_null() {
        let symbol = libc::dlsym(handle, name.as_ptr().cast());
        if !symbol.is_null() {
            return symbol;
        }
    }
    std::ptr::null_mut()
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn wall_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.tv_nsec as u64)
}

macro_rules! timed_call {
    ($name:literal, $count:expr, $dtype:expr, $call:expr) => {{
        let begin_ns = wall_ns();
        let started = Instant::now();
        let rc = $call;
        let end_ns = begin_ns.saturating_add(elapsed_ns(started));
        super::WRITER
            .lock()
            .record_direct_collective($name, begin_ns, end_ns, $count, $dtype);
        rc
    }};
}

#[no_mangle]
pub unsafe extern "C" fn HcclAllReduce(
    send: *mut c_void,
    recv: *mut c_void,
    count: u64,
    dtype: HcclDataType,
    op: HcclReduceOp,
    comm: HcclComm,
    stream: AclrtStream,
) -> HcclResult {
    type Fn = unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        u64,
        HcclDataType,
        HcclReduceOp,
        HcclComm,
        AclrtStream,
    ) -> HcclResult;
    static ORIGINAL: OnceLock<usize> = OnceLock::new();
    let ptr = *ORIGINAL.get_or_init(|| unsafe { resolve(b"HcclAllReduce\0") as usize });
    if ptr == 0 {
        return -1;
    }
    let original: Fn = std::mem::transmute(ptr);
    timed_call!(
        "HcclAllReduce",
        count,
        dtype,
        original(send, recv, count, dtype, op, comm, stream)
    )
}

#[no_mangle]
pub unsafe extern "C" fn HcclAllGather(
    send: *mut c_void,
    recv: *mut c_void,
    count: u64,
    dtype: HcclDataType,
    comm: HcclComm,
    stream: AclrtStream,
) -> HcclResult {
    type Fn = unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        u64,
        HcclDataType,
        HcclComm,
        AclrtStream,
    ) -> HcclResult;
    static ORIGINAL: OnceLock<usize> = OnceLock::new();
    let ptr = *ORIGINAL.get_or_init(|| unsafe { resolve(b"HcclAllGather\0") as usize });
    if ptr == 0 {
        return -1;
    }
    let original: Fn = std::mem::transmute(ptr);
    timed_call!(
        "HcclAllGather",
        count,
        dtype,
        original(send, recv, count, dtype, comm, stream)
    )
}

#[no_mangle]
pub unsafe extern "C" fn HcclReduceScatter(
    send: *mut c_void,
    recv: *mut c_void,
    count: u64,
    dtype: HcclDataType,
    op: HcclReduceOp,
    comm: HcclComm,
    stream: AclrtStream,
) -> HcclResult {
    type Fn = unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        u64,
        HcclDataType,
        HcclReduceOp,
        HcclComm,
        AclrtStream,
    ) -> HcclResult;
    static ORIGINAL: OnceLock<usize> = OnceLock::new();
    let ptr = *ORIGINAL.get_or_init(|| unsafe { resolve(b"HcclReduceScatter\0") as usize });
    if ptr == 0 {
        return -1;
    }
    let original: Fn = std::mem::transmute(ptr);
    timed_call!(
        "HcclReduceScatter",
        count,
        dtype,
        original(send, recv, count, dtype, op, comm, stream)
    )
}

#[no_mangle]
pub unsafe extern "C" fn HcclBroadcast(
    buffer: *mut c_void,
    count: u64,
    dtype: HcclDataType,
    root: u32,
    comm: HcclComm,
    stream: AclrtStream,
) -> HcclResult {
    type Fn = unsafe extern "C" fn(
        *mut c_void,
        u64,
        HcclDataType,
        u32,
        HcclComm,
        AclrtStream,
    ) -> HcclResult;
    static ORIGINAL: OnceLock<usize> = OnceLock::new();
    let ptr = *ORIGINAL.get_or_init(|| unsafe { resolve(b"HcclBroadcast\0") as usize });
    if ptr == 0 {
        return -1;
    }
    let original: Fn = std::mem::transmute(ptr);
    timed_call!(
        "HcclBroadcast",
        count,
        dtype,
        original(buffer, count, dtype, root, comm, stream)
    )
}
