//! Rust library for building smart contracts on the [Internet Computer].
//! More specifically it is used by [Spinner.Cash], a decentralized layer-2 protocol enabling private transactions for ICP and BTC.
//!
//! * [x] [Stable storage](storage::StableStorage).
//! * [x] [Append-only log](log::Log) using stable storage.
//! * [x] [Flat merkle tree](flat_tree::FlatTree) that is well suited for stable storage.
//! * [ ] Even queue.
//!
//! All source code are original and released under GPLv3.
//! Please make sure you understand the requirement and risk before using them in your own projects.
//!
//! [Internet Computer]: https://wiki.internetcomputer.org
//! [Spinner.Cash]: https://github.com/spinner-cash

pub mod flat_tree;
pub mod log;
pub mod storage;
