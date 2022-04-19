//! Append-only log keeping log entries in stable memory.
//!
//! Log entries are given an index, allowing for random read access.
//!
//! Internally, logs are organized in buckets so that the random read access can be made more efficiently.
//! We can first seek to the right bucket, then seek to an index within the bucket.
//! During writes, if the current bucket is full, a new bucket will be created to store the next entry.
use crate::storage::*;
use ic_cdk::export::candid::CandidType;
use serde::{Deserialize, Serialize};

/// The state and configuration of the log.
#[derive(CandidType, Serialize, Deserialize, Clone, Debug, Default)]
pub struct LogState {
    /// Stores the offset (starting position) to each buckets.
    buckets: Vec<Offset>,
    /// Number of entries in the log.
    pub size: u64,
    /// Number of entries in each bucket (must not be changed once initialized).
    pub bucket_size: usize,
    /// Max number of buckets, once reached the log is full and any writes will be rejected.
    pub max_buckets: usize,
    /// Next writing position in the storage.
    pub offset: Offset,
}

/// A generic log parameterized by its storage type `S` and log entry type `T`.
pub struct Log<'a, S, T> {
    pub state: &'a mut LogState,
    storage: &'a mut S,
    element: std::marker::PhantomData<T>,
}

impl LogState {
    /// Return an empty `LogState` with initial memory `offset`, `bucket_size` and `max_buckets` settings.
    pub fn new(offset: Offset, bucket_size: usize, max_buckets: usize) -> Self {
        LogState {
            buckets: Vec::new(),
            size: 0,
            bucket_size,
            max_buckets,
            offset,
        }
    }
}

impl<'a, S: StorageStack, T> Log<'a, S, T> {
    /// Return `Log` by initializing it with a [LogState] and a [StorageStack].
    pub fn new(state: &'a mut LogState, storage: &'a mut S) -> Self {
        Self {
            state,
            storage,
            element: std::marker::PhantomData,
        }
    }

    /// Add a new entry to the log.
    /// Return `None` if the log is full, or storage is full.
    pub fn push(&mut self, entry: T) -> Option<()>
    where
        T: candid::utils::ArgumentEncoder,
    {
        let state = &mut self.state;
        let mut storage = self.storage.new_with(state.offset);
        storage.push(entry).ok()?;
        if state.size >= state.bucket_size as u64 * state.buckets.len() as u64 {
            if state.buckets.len() >= state.max_buckets {
                return None;
            }
            state.buckets.push(state.offset);
        }
        state.size += 1;
        state.offset = storage.offset();
        Some(())
    }

    /// Return number of entries in the log.
    pub fn size(&self) -> u64 {
        self.state.size
    }

    /// Look up a log entry by index.
    /// Return `None` if the index is out of range, or it fails to read the entry.
    pub fn get(&self, index: u64) -> Option<T>
    where
        T: for<'de> candid::utils::ArgumentDecoder<'de>,
    {
        let state = &self.state;
        if index >= state.size {
            return None;
        }
        let i = (index / state.bucket_size as u64) as usize;
        assert!(i < state.buckets.len());
        let mut n;
        let offset;
        if i == state.buckets.len() - 1 {
            n = state.size;
            offset = self.state.offset;
        } else {
            n = ((i + 1) * state.bucket_size) as u64;
            offset = state.buckets[i + 1];
        }
        let mut storage = self.storage.new_with(offset);
        while n > index + 1 {
            storage.seek_prev().ok()?;
            n -= 1;
        }
        storage.pop().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::test::Stack;

    #[test]
    fn test_log_int() {
        let mut state: LogState = LogState::new(0, 3, 0);
        let mut stack = Stack::default();
        let mut log: Log<Stack, (u8,)> = Log::new(&mut state, &mut stack);
        assert_eq!(log.size(), 0);
        assert!(log.push((0,)).is_none());
        assert!(log.get(0).is_none());

        let mut state: LogState = LogState::new(0, 4, 2);
        let mut stack = Stack::default();
        let mut log: Log<Stack, (u8,)> = Log::new(&mut state, &mut stack);
        for i in 0..8 {
            assert!(log.push((i,)).is_some());
            assert_eq!(log.get(i as u64), Some((i,)));
        }
        assert_eq!(log.size(), 8);
        for i in 0..8 {
            assert_eq!(log.get(i as u64), Some((i,)));
        }
        assert_eq!(log.size(), 8);
        assert!(log.push((0,)).is_none());
    }

    #[test]
    fn test_log_str() {
        let mut state: LogState = LogState::new(0, 4, 100);
        let mut stack = Stack::default();
        let mut log: Log<Stack, (String,)> = Log::new(&mut state, &mut stack);
        for i in 0..108 {
            assert!(log.push((format!("{}", i),)).is_some());
        }
        assert_eq!(log.size(), 108);
        for i in 0..108 {
            assert_eq!(log.get(i as u64), Some((format!("{}", i),)));
        }
        assert_eq!(log.size(), 108);
    }
}
