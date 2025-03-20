/// An event queue ordered by expiry time.
///
/// Each event has an id and status, and they can be inserted/looked up/purged from the queue.
use candid::CandidType;
use serde::{Deserialize, Serialize};
use std::collections::{btree_map::Entry, BTreeMap};
use std::{error, fmt};

#[derive(
    CandidType, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Hash,
)]
/// The `nonce` field of an `EventId` must be unique, and is usually randomly generated.
pub struct EventId<I> {
    /// Expiration time since UNIX epoch (in nanoseconds).
    pub expiry: u64,
    /// Unique identifier of an event.
    pub nonce: I,
}

impl<I> EventId<I> {
    pub fn new(expiry: u64, nonce: I) -> Self {
        Self { expiry, nonce }
    }
}

#[derive(Clone, Debug)]
/// Error when an operation cannot be performed because the event queue reaches its max capacity.
pub struct EventQueueFullError;

impl error::Error for EventQueueFullError {}

impl fmt::Display for EventQueueFullError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Event queue is full")
    }
}

#[derive(CandidType, Serialize, Deserialize, Clone, Debug, Hash)]
/// An event queue is an ordered list of events of nonce type `I` and status type `T`.
/// Its max capacity is fixed at initialization.
pub struct EventQueue<I: Ord, T> {
    capacity: usize,
    events: BTreeMap<EventId<I>, T>,
    index: BTreeMap<I, EventId<I>>,
}

impl<I: Ord, T> Default for EventQueue<I, T> {
    /// Return an empty event queue of zero capacity.
    fn default() -> Self {
        EventQueue::new(0)
    }
}

fn is_vacant<K, V>(entry: &Entry<K, V>) -> bool {
    matches!(entry, Entry::Vacant(_))
}

impl<I: Ord, T> EventQueue<I, T> {
    /// Return an event queue of the given max capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            events: BTreeMap::default(),
            index: BTreeMap::default(),
        }
    }

    /// Insert an event of the given id and status into the queue.
    /// Return error if the queue is full.
    /// This operation is `O(log(n))`.
    pub fn insert(&mut self, id: EventId<I>, event: T) -> Result<(), EventQueueFullError>
    where
        I: Clone,
    {
        let size = self.events.len();
        let entry = self.events.entry(id.clone());
        if is_vacant(&entry) {
            if size >= self.capacity {
                return Err(EventQueueFullError);
            }
            entry.or_insert(event);
            self.index.insert(id.nonce.clone(), id);
        } else {
            entry.and_modify(|v| *v = event);
        }
        Ok(())
    }

    /// Remove and return the first matching event.
    /// This operation is linear in the events it has to search through.
    pub fn pop_front<F: Fn(&T) -> bool>(&mut self, matching: F) -> Option<(EventId<I>, T)>
    where
        I: Clone,
    {
        if let Some(id) =
            self.events
                .iter()
                .find_map(|(id, e)| if matching(e) { Some(id.clone()) } else { None })
        {
            self.index.remove(&id.nonce);
            self.events.remove(&id).map(|e| (id.clone(), e))
        } else {
            None
        }
    }

    /// Remove all events of the matching status with an expiry less than the given expiry.
    /// This operation is in the number of events matching the expiry condition.
    pub fn purge<F: Fn(&T) -> bool>(&mut self, expiry: u64, matching: F)
    where
        I: Default,
    {
        let key = EventId {
            expiry,
            nonce: I::default(),
        };
        let mut newer_events = self.events.split_off(&key);
        self.events.retain(|k, v| {
            let to_remove = matching(v);
            if to_remove {
                self.index.remove(&k.nonce);
            }
            !to_remove
        });
        self.events.append(&mut newer_events);
    }

    /// Remove all events with an expiry less than the given expiry.
    /// This operation is `O(log(n))+O(m)`, where `m` is the number of events removed.
    pub fn purge_expired(&mut self, expiry: u64)
    where
        I: Default,
    {
        let key = EventId {
            expiry,
            nonce: I::default(),
        };
        let to_keep = self.events.split_off(&key);
        for to_remove in self.events.keys() {
            self.index.remove(&to_remove.nonce);
        }
        self.events = to_keep;
    }

    /// Lookup event status given an event id.
    /// This operation is `O(log(n))`.
    pub fn get(&self, id: &EventId<I>) -> Option<&T> {
        self.events.get(id)
    }

    /// Lookup event status given an event id.
    /// This operation is `O(log(n))`.
    pub fn remove(&mut self, id: &EventId<I>) -> Option<T> {
        if let Some(value) = self.events.remove(id) {
            self.index.remove(&id.nonce);
            Some(value)
        } else {
            None
        }
    }

    /// Modify the status of an event given its id.
    /// This operation is `O(log(n))`.
    pub fn modify<F: FnOnce(&mut T)>(&mut self, id: EventId<I>, f: F) {
        self.events.entry(id).and_modify(|v: &mut T| f(v));
    }

    /// Return both an event id and its status if the given nonce is found.
    /// This operation is `O(log(n))`.
    pub fn find(&self, nonce: &I) -> Option<(&EventId<I>, &T)> {
        self.index
            .get(nonce)
            .and_then(|id| self.events.get(id).map(|e| (id, e)))
    }

    /// Return max capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return number of events in the queue.
    /// This operation is constant time.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return true if the queue is at its max capacity.
    /// This operation is constant time.
    pub fn is_full(&self) -> bool {
        self.events.len() == self.capacity
    }

    /// Return true if the queue is empty.
    /// This operation is constant time.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Return an iterator of all events in the order of their `EventId`, i.e., ordered by expiry first, then by nonce.
    pub fn iter(&self) -> Box<dyn Iterator<Item = (&EventId<I>, &T)> + '_> {
        Box::new(self.events.iter())
    }

    /// Check consistency of the event queue.
    /// It should always return true if the implementation is correct.
    #[cfg(test)]
    fn selfcheck(&self) -> bool
    where
        I: Eq + Clone,
    {
        use std::collections::BTreeSet;
        self.len() <= self.capacity
            && self.events.len() == self.index.len()
            && self.events.keys().collect::<Vec<_>>() == self.index.values().collect::<Vec<_>>()
            && self.index.keys().cloned().collect::<BTreeSet<_>>().len() == self.index.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use assert_matches::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Item {
        Begin(u8),
        Middle(u8),
        End(u8),
    }
    use Item::*;

    fn is_begin(item: &Item) -> bool {
        matches!(item, Begin(_))
    }

    fn is_middle(item: &Item) -> bool {
        matches!(item, Middle(_))
    }

    fn is_end(item: &Item) -> bool {
        matches!(item, End(_))
    }

    #[test]
    fn basic() {
        let mut q: EventQueue<u8, Item> = EventQueue::new(10);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert!(q.selfcheck());

        for i in 0..10 {
            let expiry = if i < 5 { 0 } else { 1 };
            assert_matches!(q.insert(EventId::new(expiry, i), Begin(i)), Ok(_));
            assert!(q.selfcheck());
        }
        // cannot insert any more
        assert_matches!(
            q.insert(EventId::new(2, 0), Begin(0)),
            Err(EventQueueFullError)
        );
        assert!(q.selfcheck());
        // still allow insertion of existing key
        assert_matches!(q.insert(EventId::new(0, 1), Middle(10)), Ok(_));
        assert_matches!(q.insert(EventId::new(0, 4), End(40)), Ok(_));
        assert!(q.selfcheck());

        // confirm if everything is in there
        for i in 0..10 {
            assert_matches!(q.find(&i), Some(_));
        }

        // test pop_front
        assert_eq!(q.pop_front(is_begin), Some((EventId::new(0, 0), Begin(0))));
        assert!(q.selfcheck());
        assert_eq!(q.pop_front(is_begin), Some((EventId::new(0, 2), Begin(2))));
        assert!(q.selfcheck());
        assert_eq!(
            q.pop_front(is_middle),
            Some((EventId::new(0, 1), Middle(10)))
        );
        assert!(q.selfcheck());
        assert_eq!(q.len(), 7);

        let mut p = q.clone();

        // test purge
        q.purge(1, is_begin);
        assert_eq!(q.len(), 6);
        assert!(q.selfcheck());
        assert_eq!(q.pop_front(is_begin), Some((EventId::new(1, 5), Begin(5))));
        assert!(q.selfcheck());
        assert_eq!(q.pop_front(is_end), Some((EventId::new(0, 4), End(40))));
        assert!(q.selfcheck());
        assert_eq!(q.len(), 4);

        q.purge(2, is_begin);
        assert_eq!(q.len(), 0);
        assert!(q.selfcheck());

        // test purge_expired
        println!("{:?}", p.iter().collect::<Vec<_>>());
        p.purge_expired(1);
        assert_eq!(p.len(), 5);
        assert!(p.selfcheck());

        p.purge_expired(2);
        assert_eq!(p.len(), 0);
        assert!(p.selfcheck());
    }
}
