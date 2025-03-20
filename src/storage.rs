//! Utilities to help manage canister stable memory.
//!
//! [ic-cdk] gives only low-level interface to canister stable memory, with no memory manager or allocator.
//!
//! To help with easy access:
//!
//! * [StableStorage] supports sequential read & write of bytes by implementing [io::Read] and [io::Write] traits.
//!
//! * [StorageStack] provides a stack interface allowing arbitrary values to be pushed onto and popped off the stable memory.
//!   [StableStorage] implements this trait.
//!   Being a trait it allows alternative implementations, for example in testing code.
//!
//! [ic-cdk]: https://docs.rs/ic-cdk/latest
use ic_cdk::api::stable;
use std::{error, fmt, io};

/// Possible errors when dealing with stable memory.
#[derive(Debug)]
pub enum StorageError {
    /// No more stable memory could be allocated.
    OutOfMemory,
    /// Attempted to read more stable memory than had been allocated.
    OutOfBounds,
    /// Candid encoding error.
    Candid(candid::Error),
}

impl From<candid::Error> for StorageError {
    fn from(err: candid::Error) -> StorageError {
        StorageError::Candid(err)
    }
}

impl From<StorageError> for io::Error {
    fn from(err: StorageError) -> io::Error {
        match err {
            StorageError::Candid(err) => io::Error::new(io::ErrorKind::Other, err),
            err => io::Error::new(io::ErrorKind::OutOfMemory, err),
        }
    }
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::OutOfMemory => f.write_str("Out of memory"),
            Self::OutOfBounds => f.write_str("Read exceeds allocated memory"),
            Self::Candid(err) => write!(f, "{}", err),
        }
    }
}

impl error::Error for StorageError {}

/// Representation of a memory address.
pub type Offset = u64;

/// Reader/Writer of the canister stable memory.
///
/// It keeps track of the current read/write offset, and will attempt to grow the stable memory as needed.
pub struct StableStorage {
    /// Offset of the next read or write.
    pub offset: Offset,
    /// Current capacity, as in number of pages.
    capacity: u64,
}

/// The default instance of `StableStorage` starts at offset 0 if the current stable memory capacity is 0 (which means it is never used).
///
/// Otherwise it reads the offset value from the last 8 bytes (in little endian) of the stable memory.
impl Default for StableStorage {
    fn default() -> Self {
        let mut storage = Self {
            offset: 0,
            capacity: stable::stable_size(),
        };
        if storage.capacity > 0 {
            let cap = storage.capacity << 16;
            let mut bytes = [0; 8];
            stable::stable_read(cap - 8, &mut bytes);
            storage.offset = u64::from_le_bytes(bytes);
        }
        storage
    }
}

impl StableStorage {
    /// Attempt to grow the memory by adding new pages.
    fn grow(&mut self, added_pages: u64) -> Result<(), StorageError> {
        let old_page_count =
            stable::stable_grow(added_pages).map_err(|_| StorageError::OutOfMemory)?;
        self.capacity = old_page_count + added_pages;
        Ok(())
    }

    /// Create a new instance of [StableStorage].
    pub fn new() -> Self {
        Default::default()
    }

    /// Write current offset value to the last 8 bytes (in little-endian) of the stable memory.
    /// This is an important step if you plan to later resume by reconstructing `StableStorage` from the stable memory.
    pub fn finalize(mut self) -> Result<(), io::Error> {
        let mut cap = self.capacity << 16;
        if self.offset + 8 > cap {
            self.grow(1)?;
            cap = self.capacity << 16;
        }
        let bytes = self.offset.to_le_bytes();
        io::Write::write(&mut self, &bytes)?;
        stable::stable_write(cap - 8, &bytes);
        Ok(())
    }
}

impl io::Write for StableStorage {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        if self.offset + buf.len() as u64 > (self.capacity << 16) {
            self.grow((buf.len() >> 16) as u64 + 1)?
        }

        stable::stable_write(self.offset, buf);
        self.offset += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        Ok(())
    }
}

/// Stack interface for stable memory that supports push and pop of arbitrary values that implement the [Candid] interface.
///
/// [Candid]: https://docs.rs/candid/latest
pub trait StorageStack {
    /// Return a new [StorageStack] object with the given offset.
    fn new_with(&self, offset: Offset) -> Self;

    /// Return the current read/write offset.
    fn offset(&self) -> Offset;

    /// Push a value to the end of the stack.
    fn push<T>(&mut self, t: T) -> Result<(), io::Error>
    where
        T: candid::utils::ArgumentEncoder;

    /// Pop a value from the end of the stack.
    /// In case of `OutOfBounds` error, offset is not changed.
    /// In case of Candid decoding error, offset may be changed.
    fn pop<T>(&mut self) -> Result<T, io::Error>
    where
        T: for<'de> candid::utils::ArgumentDecoder<'de>;

    /// Seek to the start of previous value by changing the offset.
    /// This is similar to `pop` but without reading the actual value.
    fn seek_prev(&mut self) -> Result<(), io::Error>;
}

impl StorageStack for StableStorage {
    fn new_with(&self, offset: Offset) -> Self {
        Self {
            offset,
            capacity: self.capacity,
        }
    }

    fn offset(&self) -> Offset {
        self.offset
    }

    fn push<T>(&mut self, t: T) -> Result<(), io::Error>
    where
        T: candid::utils::ArgumentEncoder,
    {
        let prev_offset = self.offset;
        candid::write_args(self, t).map_err(StorageError::from)?;
        let bytes = prev_offset.to_le_bytes();
        io::Write::write(self, &bytes)?;
        Ok(())
    }

    fn pop<T>(&mut self) -> Result<T, io::Error>
    where
        T: for<'de> candid::utils::ArgumentDecoder<'de>,
    {
        let end = self.offset - 8;
        self.seek_prev()?;
        let size = (end - self.offset) as usize;
        let mut bytes = vec![0; size];
        stable::stable_read(self.offset, &mut bytes);
        let mut de = candid::de::IDLDeserialize::new(&bytes).map_err(StorageError::Candid)?;
        let res = candid::utils::ArgumentDecoder::decode(&mut de).map_err(StorageError::Candid)?;
        Ok(res)
    }

    fn seek_prev(&mut self) -> Result<(), io::Error> {
        if self.offset < 8 {
            return Err(StorageError::OutOfBounds.into());
        }
        let mut bytes = [0; 8];
        let end = self.offset - 8;
        stable::stable_read(end, &mut bytes);
        let start = u64::from_le_bytes(bytes);
        if start > end {
            return Err(StorageError::OutOfBounds.into());
        }
        self.offset = start;
        Ok(())
    }
}

impl io::Read for StableStorage {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, io::Error> {
        let cap = self.capacity << 16;
        let read_buf = if buf.len() as u64 + self.offset > cap {
            if self.offset < cap {
                &mut buf[..(cap - self.offset) as usize]
            } else {
                return Err(StorageError::OutOfBounds.into());
            }
        } else {
            buf
        };
        stable::stable_read(self.offset, read_buf);
        self.offset += read_buf.len() as u64;
        Ok(read_buf.len())
    }
}

pub mod test {
    use super::*;
    use candid::encode_args;
    use std::cell::RefCell;
    use std::io;
    use std::rc::Rc;

    /// A vector-based implementation of [StorageStack], used for testing purpose.
    #[derive(Clone, Default)]
    pub struct Stack {
        stack: Rc<RefCell<Vec<Vec<u8>>>>,
        offset: Offset,
        index: usize,
    }

    impl StorageStack for Stack {
        fn new_with(&self, offset: Offset) -> Stack {
            let mut s = 0;
            let mut index = 0;
            while s < offset {
                s += self.stack.as_ref().borrow()[index].len() as Offset;
                index += 1;
            }
            Stack {
                stack: Rc::clone(&self.stack),
                offset,
                index,
            }
        }

        fn offset(&self) -> Offset {
            self.offset
        }

        /// Save a value to the end of stable memory.
        fn push<T>(&mut self, t: T) -> Result<(), io::Error>
        where
            T: candid::utils::ArgumentEncoder,
        {
            let bytes: Vec<u8> = encode_args(t).unwrap();
            self.offset += bytes.len() as Offset;
            let mut stack = self.stack.borrow_mut();
            if stack.len() > self.index {
                stack[self.index] = bytes;
            } else {
                stack.push(bytes)
            }
            self.index += 1;
            Ok(())
        }

        /// Pop a value from the end of stable memory.
        /// In case of `OutOfBounds` error, offset is not changed.
        /// In case of Candid decoding error, offset will be changed anyway.
        fn pop<T>(&mut self) -> Result<T, io::Error>
        where
            T: for<'de> candid::utils::ArgumentDecoder<'de>,
        {
            self.seek_prev()?;
            let bytes = self.stack.borrow()[self.index].clone();
            let mut de = candid::de::IDLDeserialize::new(&bytes).unwrap();
            Ok(candid::utils::ArgumentDecoder::decode(&mut de).unwrap())
        }

        fn seek_prev(&mut self) -> Result<(), io::Error> {
            assert!(self.index > 0);
            let bytes = self.stack.borrow()[self.index - 1].clone();
            self.index -= 1;
            assert!(self.offset >= bytes.len() as Offset);
            self.offset -= bytes.len() as Offset;
            Ok(())
        }
    }
}
