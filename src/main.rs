use core::hint::black_box;
use dsi_progress_logger::no_logging;
use gxhash::HashMap;
use std::time::Instant;
use sux::{func::VBuilder, utils::FromIntoIterator};

use clap::Parser;
use ph::phast;
use ptr_hash::{PtrHash, PtrHashParams};
use rand::{rngs::SmallRng, seq::SliceRandom, RngCore, SeedableRng};

#[derive(Parser, Debug)]
#[command(about = "Build contributor and origin collaboration graphs")]
struct Args {
    n: usize,
}

fn main() {
    let args = Args::parse();
    let mut r = SmallRng::seed_from_u64(0);

    let n = args.n;

    let mut keys = Vec::with_capacity(n);
    keys.resize_with(n, || r.next_u64());

    // GxHashMap
    let mut map = HashMap::with_capacity_and_hasher(n, Default::default());
    keys.iter().for_each(|&k| {
        map.insert(k, k);
    });
    keys.shuffle(&mut r);

    let now = Instant::now();
    for i in 0..n {
        let _: _ = black_box(*unsafe { map.get(&keys[i]).unwrap_unchecked() });
    }
    println!(
        "GxHM             lookup time {:8.3}ns",
        now.elapsed().as_nanos() as f64 / n as f64
    );

    // PtrHash

    let ptr_hash = <PtrHash>::new(keys.as_slice(), PtrHashParams::default());
    let mut values = vec![0; keys.len()];
    map.iter().for_each(|(k, v)| {
        values[ptr_hash.index(k)] = *v;
    });

    let now = Instant::now();
    for i in 0..n {
        debug_assert_eq!(
            values[ptr_hash.index(&keys[i])],
            *map.get(&keys[i]).unwrap()
        );
        _ = black_box(values[ptr_hash.index(&keys[i])]);
    }
    println!(
        "PtrHash          lookup time {:8.3}ns",
        now.elapsed().as_nanos() as f64 / n as f64
    );

    let now = Instant::now();
    index_all_with_prefetch(&values, keys.iter().map(|key| ptr_hash.index(key)));
    println!(
        "PtrHash +pf vals lookup time {:8.3}ns",
        now.elapsed().as_nanos() as f64 / n as f64
    );

    let now = Instant::now();
    let indices = ptr_hash.index_stream::<32, true, _>(&keys);
    indices.for_each(|idx| {
        _ = black_box(values[idx]);
    });
    println!(
        "PtrHash +pf idx  lookup time {:8.3}ns",
        now.elapsed().as_nanos() as f64 / n as f64
    );

    let now = Instant::now();
    index_all_with_prefetch(&values, ptr_hash.index_stream::<32, true, _>(&keys));
    println!(
        "PtrHash +pf both lookup time {:8.3}ns",
        now.elapsed().as_nanos() as f64 / n as f64
    );

    drop(ptr_hash);

    // Phast

    let phast = phast::Function2::from_slice_mt(&keys);

    let mut values = vec![0; keys.len()];
    map.iter().for_each(|(k, v)| {
        values[phast.get(k) as usize] = *v;
    });

    let now = Instant::now();
    for i in 0..n {
        debug_assert_eq!(
            values[phast.get(&keys[i]) as usize],
            *map.get(&keys[i]).unwrap()
        );
        _ = black_box(values[phast.get(&keys[i])]);
    }
    println!(
        "Phast            lookup time {:8.3}ns",
        now.elapsed().as_nanos() as f64 / n as f64
    );

    let now = Instant::now();
    index_all_with_prefetch(&values, keys.iter().map(|key| phast.get(key)));
    println!(
        "Phast +pf vals   lookup time {:8.3}ns",
        now.elapsed().as_nanos() as f64 / n as f64
    );

    drop(phast);

    let vfunc = VBuilder::<_, Box<[_]>, _>::default()
        .try_build_func(
            FromIntoIterator::from(keys.clone()),
            FromIntoIterator::from(keys.clone()),
            no_logging![],
        )
        .unwrap();

    let now = Instant::now();
    for i in 0..n {
        debug_assert_eq!(vfunc.get(&keys[i]), *map.get(&keys[i]).unwrap());
        _ = black_box(vfunc.get(&keys[i]));
    }
    println!(
        "VFunc            lookup time {:8.3}ns",
        now.elapsed().as_nanos() as f64 / n as f64
    );
}

#[inline(always)]
fn index_all_with_prefetch(values: &[u64], indices: impl IntoIterator<Item = usize>) {
    let mut buf = [0usize; 32];
    let mut i = 0;
    indices.into_iter().for_each(|idx| {
        _ = black_box(values[buf[i % 32]]);
        buf[i % 32] = idx;
        prefetch_index(&values, idx);
        i += 1;
    });
}

/// Prefetch the given cacheline into L1 cache.
pub(crate) fn prefetch_index<T>(s: &[T], index: usize) {
    let ptr = s.as_ptr().wrapping_add(index) as *const u64;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // TODO: Put this behind a feature flag.
        // std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_LOCALITY3);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        // Do nothing.
    }
}
