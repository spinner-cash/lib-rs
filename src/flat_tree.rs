//! Merkle Tree implementation based on Flat In-Order Tree.
//!
//! Flat In-Order Tree is described in DEP-0002:
//! <https://github.com/datprotocol/DEPs/blob/master/proposals/0002-hypercore.md>
//!
//! A Flat In-Order Tree represents a binary tree as a list, also known as Bin Numbers defined in PPSP RFC 7574:
//! <https://datatracker.ietf.org/doc/rfc7574/>
//!
//! We assume the max node index is 63-bit, which limits the max tree depth (level) to 62.
//!
//! A merkle tree with flat layout is storage friendly (i.e. no fragmentation and grows linearly) with O(1) leaf and node access.
//! It stores all leave nodes, caches the hash of intermediate nodes, and provides efficient merkle path computation for verification purposes.

/// Abstraction over the underlying storage to allow flexible choice of implementation.
/// For example, an implementation may store multiple trees in a column format, while another may overlay in-memory cache on top of external storage.
pub trait FlatStore<T> {
    /// Writes a value at the given offset.
    ///
    /// Depending on the implementation, it may panic or silently fail if out of space.
    /// It is up to the caller to check these conditions before calling this function.
    fn write(&self, offset: u64, value: &T);

    /// Reads a value off the given offset.
    ///
    /// Depending on the implementation, it may not be able to read any value out.
    /// It is up to the caller to prevent errors (e.g. out-of-bound before calling this function.
    fn read(&self, offset: u64, value: &mut T);
}

/// Abstraction over binary hash operation.
/// The value of a merkle hash is computed by hashing its left branch and right branch.
/// By default it assumes a value of "zero" if a branch does not exist.
/// The actual hashing algorithm and the value of zeros at a given tree level are implementation dependent.
pub trait Merkleable {
    /// Return the zero at a given level (leaf level is 0).
    fn zeros(level: usize) -> Self;

    /// Return the hash of self with another.
    fn hash(&self, another: &Self) -> Self;
}

/// Representation of a flat tree of element type `T` where the `'a` is the life-time parameter of the underlying [FlatStore].
///
/// At the moment the tree is append only, i.e. it can add new leave elements but not update or deletion.
pub struct FlatTree<'a, T> {
    store: &'a dyn FlatStore<T>,
    // Total number of elements, including both leaves and internal nodes.
    // Number of leaves is exactly size / 2.
    size: u64,
}

/// Helper to iterate from a leaf node to one of the roots.
pub struct LocalPathIterator<'a, T> {
    tree: &'a FlatTree<'a, T>,
    cursor: Option<(Index, Pos<T>)>,
    root_indices: Vec<Index>,
}

/// Position of a node is either a leaf, a root, or a left or right node.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Pos<T> {
    Start(T),
    Left(T),
    Right(T),
}

impl<'a, T: Merkleable> Iterator for LocalPathIterator<'a, T> {
    type Item = Pos<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut cursor = None;
        std::mem::swap(&mut cursor, &mut self.cursor);
        match cursor {
            Some((Index(i), value)) => {
                let Index(p) = parent_of(Index(i));
                if self.root_indices.iter().any(|x| *x == Index(i)) {
                    self.cursor = None;
                } else {
                    //let parent = self.tree.read(p);
                    let j = p * 2 - i;
                    let v = self.tree.read(j);
                    self.cursor =
                        Some((Index(p), if j < i { Pos::Left(v) } else { Pos::Right(v) }));
                }
                Some(value)
            }
            None => None,
        }
    }
}

/// Helper to iterate from a leaf node to the top root (at a level that is equal to full tree depth).
///
/// Note that nodes in this path may not exist in the store and will have to be calculated.
pub struct FullPathIterator<'a, T> {
    tree: &'a FlatTree<'a, T>,
    cursor: Option<(Index, Pos<T>)>,
    roots: Vec<(Index, T)>,
}

impl<'a, T: Merkleable> Iterator for FullPathIterator<'a, T> {
    type Item = Pos<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut cursor = None;
        std::mem::swap(&mut cursor, &mut self.cursor);
        match cursor {
            Some((Index(i), value)) => {
                let Index(p) = parent_of(Index(i));
                if Index(i) == self.roots[0].0 {
                    self.cursor = None;
                } else {
                    let j = 2 * p - i;
                    let hash = if let Some(k) = self.roots.iter().position(|(x, _)| *x == Index(j))
                    {
                        let (_, hash) = self.roots.remove(k);
                        hash
                    } else if j >= self.tree.size {
                        let Level(l) = level_of(Index(j));
                        T::zeros(l)
                    } else {
                        self.tree.read(j)
                    };
                    self.cursor = Some((
                        Index(p),
                        if j < i {
                            Pos::Left(hash)
                        } else {
                            Pos::Right(hash)
                        },
                    ));
                }
                Some(value)
            }
            None => None,
        }
    }
}

impl<'a, T: Merkleable> FlatTree<'a, T> {
    /// Return a [FlatTree] object of given size with the underlying [FlatStore] as storage.
    /// If a non-zero size is given, it assumes a valid tree has been previously created on the same store.
    pub fn new(store: &'a dyn FlatStore<T>, size: u64) -> Self {
        Self { store, size }
    }

    /// Depth of a full tree.
    pub fn depth(&self) -> usize {
        if self.size == 0 {
            return 0;
        }
        let w = self.size;
        let mut n = 0;
        let mut i = 1;
        while w > i {
            i *= 2;
            n += 1;
        }
        n
    }

    /// Width of the tree (number of leave nodes).
    pub fn width(&self) -> u64 {
        self.size / 2
    }

    /// Return true if tree is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Number of leaves in the tree (same as width).
    pub fn len(&self) -> u64 {
        self.size / 2
    }

    /// Append a leaf value to the tree.
    pub fn push(&mut self, mut value: T) {
        let mut i = self.size;
        self.store.write(i, &value);
        let Level(l) = level_of(Index(i + 1));
        self.store.write(i + 1, &T::zeros(l));
        self.size += 2;
        let mut level = 0;
        loop {
            let Index(p) = parent_of(Index(i));
            let j = p * 2 - i;
            if p >= self.size || j >= self.size {
                break;
            }
            let mut sibling = T::zeros(level);
            self.store.read(j, &mut sibling);
            value = if i < j {
                value.hash(&sibling)
            } else {
                sibling.hash(&value)
            };
            self.store.write(p, &value);
            i = p;
            level += 1;
        }
    }

    /// Append a number of leaf values to the tree.
    pub fn append(&mut self, value: Vec<T>) {
        for v in value.into_iter() {
            self.push(v)
        }
    }

    /// Return the value at a given leaf index if it exists.
    pub fn get(&self, leaf_index: u64) -> Option<T> {
        if leaf_index >= self.len() {
            None
        } else {
            Some(self.read(leaf_index * 2))
        }
    }

    /// Iterate over all leaf values.
    pub fn iter(&self) -> Box<dyn Iterator<Item = T> + '_> {
        Box::new((0..self.len()).map(|i| self.get(i).unwrap()))
    }

    /// Return the top root of the full tree.
    pub fn root(&self) -> T {
        self.full_roots()
            .into_iter()
            .next()
            .map(|x| x.1)
            .unwrap_or_else(|| T::zeros(0))
    }

    #[cfg(test)]
    /// Return a list of sub-tree roots.
    fn local_roots(&self) -> Vec<T> {
        roots_of(Width(self.width()))
            .into_iter()
            .map(|Index(i)| self.read(i))
            .collect()
    }

    /// Return the full set of roots, arranged in descending order of depth, where the previous one is the parent of the next.
    fn full_roots(&self) -> Vec<(Index, T)> {
        let root_indices = roots_of(Width(self.width()));
        let mut roots = Vec::new();
        if root_indices.is_empty() {
            return roots;
        };
        let mut n = root_indices.len() - 1;
        let Index(mut i) = root_indices[n];
        let mut hash = self.read(i);
        let Level(mut level) = level_of(Index(i));
        while n > 0 {
            let Index(p) = parent_of(Index(i));
            let j = p * 2 - i;
            let sibling = if n > 0 && Index(j) == root_indices[n - 1] {
                n -= 1;
                self.read(j)
            } else {
                T::zeros(level)
            };
            let new_hash = if i < j {
                hash.hash(&sibling)
            } else {
                sibling.hash(&hash)
            };
            roots.push((Index(i), hash));
            hash = new_hash;
            i = p;
            level += 1;
        }
        roots.push((Index(i), hash));
        roots.reverse();
        roots
    }

    /// Return the merkle path iterating from the leaf at the given index to one of the local roots.
    pub fn local_path(&self, i: u64) -> LocalPathIterator<'_, T> {
        let cursor = self.get(i).map(|value| (Index(i * 2), Pos::Start(value)));
        LocalPathIterator {
            tree: self,
            cursor,
            root_indices: roots_of(Width(self.width())),
        }
    }

    /// Return the merkle path iterating from the leaf at the given index to top root.
    pub fn full_path(&self, i: u64) -> FullPathIterator<'_, T> {
        let cursor = self.get(i).map(|value| (Index(i * 2), Pos::Start(value)));
        FullPathIterator {
            tree: self,
            cursor,
            roots: self.full_roots(),
        }
    }

    /// Return the full path iterating from the given node index (either leaf or branch node) to top root.
    /// Note that for leaf nodes, `node_index =  leaf_index * 2`.
    pub fn full_path_from_node_index(&self, i: u64) -> FullPathIterator<'_, T>
    where
        T: Clone,
    {
        let roots = self.full_roots();
        let index = Index(i);
        let mut cursor = None;
        if inside_tree(index, Width(self.width())) {
            cursor = Some((index, Pos::Start(self.read(i))))
        }
        if cursor.is_none() {
            cursor = roots.iter().find_map(|(j, r)| {
                if index == *j {
                    Some((index, Pos::Start(r.clone())))
                } else {
                    None
                }
            });
        };
        FullPathIterator {
            tree: self,
            cursor,
            roots: self.full_roots(),
        }
    }

    // Read the value at the given node index.
    fn read(&self, i: u64) -> T {
        let Level(l) = level_of(Index(i));
        let mut x = T::zeros(l);
        self.store.read(i, &mut x);
        x
    }
}

#[cfg(test)]
fn debug_print<T: Merkleable>(tree: &FlatTree<'_, T>)
where
    T: std::fmt::Debug,
{
    let w = tree.width();
    for (i, level) in tree_of(Width(w)).into_iter().enumerate() {
        match level {
            None => println!(),
            Some(Level(l)) => {
                for _ in 0..2 * l {
                    print!(" ")
                }
                println!("{:?}", tree.read(i as u64));
            }
        }
    }
}

#[cfg(test)]
/// Return the list of levels of all indices in the tree of the given width.
/// Indices of incomplete nodes (those do not have both children) do not have a level, and its corresponding value in the returned list is `None`.
fn tree_of(w: Width) -> Vec<Option<Level>> {
    let Width(n) = w;
    let mut levels = Vec::new();
    let m = (n - 1) * 2;
    for i in 0..(m + 1) {
        let Level(l) = level_of(Index(i));
        let subtree_size = (1 << l) - 1;
        if subtree_size + i > m {
            levels.push(None)
        } else {
            levels.push(Some(Level(l)))
        }
    }
    levels
}

/// Return true if the given Index is inside the tree of given Width.
fn inside_tree(i: Index, w: Width) -> bool {
    let Width(n) = w;
    let m = (n - 1) * 2;
    let Level(l) = level_of(i);
    let subtree_size = (1 << l) - 1;
    subtree_size + i.0 <= m
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Level(usize);

/// Max tree depth on a 64-bit architecture is 62.
pub const MAX_LEVEL: usize = 62;

impl From<Level> for usize {
    fn from(l: Level) -> usize {
        l.0
    }
}

impl From<usize> for Level {
    fn from(n: usize) -> Self {
        assert!(n <= MAX_LEVEL, "level out of bound");
        Level(n)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Width(u64);

const MAX_WIDTH: u64 = 1 << MAX_LEVEL;

impl From<Width> for u64 {
    fn from(w: Width) -> u64 {
        w.0
    }
}

impl From<u64> for Width {
    fn from(n: u64) -> Self {
        assert!(n <= MAX_WIDTH, "width out of bound");
        Width(n)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Index(u64);

const MAX_ROOT_INDEX: u64 = MAX_WIDTH + 1;

const MAX_INDEX: u64 = MAX_ROOT_INDEX * 2;

impl From<Index> for u64 {
    fn from(i: Index) -> u64 {
        i.0
    }
}

impl From<u64> for Index {
    fn from(n: u64) -> Self {
        assert!(n <= MAX_INDEX, "index out of bound");
        Index(n)
    }
}

/// Return the level of the given `node_index`.
pub fn level_of_node_index(node_index: u64) -> usize {
    level_of(Index(node_index)).0
}

/// Level is equal to trailing number of 1s in the index.
fn level_of(Index(mut i): Index) -> Level {
    let mut l = 0;
    while i & 1 == 1 {
        l += 1;
        i /= 2;
    }
    Level(l)
}

/// Return the list of root indices given a tree width.
fn roots_of(Width(w): Width) -> Vec<Index> {
    let mut base = 1;
    let mut i = w * 2;
    let mut basis = Vec::new();
    while i > 0 {
        if i & 1 == 1 {
            basis.push(base)
        }
        base *= 2;
        i /= 2;
    }
    let n = basis.len();
    let mut roots = Vec::with_capacity(n);
    base = 0;
    for i in 0..n {
        let x = basis[n - i - 1];
        roots.push(Index(base + x / 2 - 1));
        base += x;
    }
    roots
}

/// Get the start index and step size at the given level, such that the sequence of indices at this level is: `[start, start + step .. ]`
fn at_level(Level(l): Level) -> (Index, u64) {
    (Index((1 << l) - 1), 1 << (l + 1))
}

/// Return the parent of the given index.
fn parent_of(index: Index) -> Index {
    let Index(i) = index;
    assert!(i != MAX_ROOT_INDEX, "max root has no parent");
    let Level(l) = level_of(index);
    let (Index(start), step) = at_level(Level(l));
    let (Index(start_), step_) = at_level(Level(l + 1));
    let pos = (i - start) / step;
    Index(start_ + step_ * (pos / 2))
}

#[cfg(test)]
/// Return the children of the given index.
fn children_of(index: Index) -> Option<(Index, Index)> {
    let Index(i) = index;
    let Level(l) = level_of(index);
    if l == 0 {
        return None;
    };
    let (Index(start), step) = at_level(Level(l));
    let (Index(start_), step_) = at_level(Level(l - 1));
    let pos = (i - start) / step;
    let left = start_ + pos * 2 * step_;
    let right = i * 2 - left;
    Some((Index(left), Index(right)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_of() {
        assert_eq!(
            tree_of(Width::from(7))
                .iter()
                .map(|x| x.map(|x| usize::from(x) as i64).unwrap_or(-1))
                .collect::<Vec<_>>(),
            [0, 1, 0, 2, 0, 1, 0, -1, 0, 1, 0, -1, 0]
        );
    }

    #[test]
    fn test_level_of() {
        assert_eq!(level_of(Index::from(23)), Level::from(3));
    }

    #[test]
    fn test_roots_of() {
        assert_eq!(
            roots_of(Width::from(7)),
            [Index::from(3), Index::from(9), Index::from(12)]
        );
    }

    #[test]
    fn test_at_level() {
        assert_eq!(at_level(Level::from(1)), (Index::from(1), 4));
        assert_eq!(at_level(Level::from(2)), (Index::from(3), 8));
    }

    #[test]
    fn test_parent_of() {
        assert_eq!(parent_of(Index::from(3)), Index::from(7));
        assert_eq!(parent_of(Index::from(23)), Index::from(15));
    }

    #[test]
    #[should_panic]
    fn test_parent_of_out_of_bound() {
        parent_of(Index::from(MAX_ROOT_INDEX));
    }

    #[test]
    fn test_children_of() {
        assert_eq!(
            children_of(Index::from(3)),
            Some((Index::from(1), Index::from(5)))
        );
        assert_eq!(children_of(Index::from(6)), None);
        assert_eq!(
            children_of(Index::from(5)),
            Some((Index::from(4), Index::from(6)))
        );
        assert_eq!(
            children_of(Index::from(9)),
            Some((Index::from(8), Index::from(10)))
        );
        assert_eq!(children_of(Index::from(12)), None);
    }

    use std::cell::RefCell;

    impl FlatStore<String> for RefCell<Vec<String>> {
        fn write(&self, offset: u64, value: &String) {
            self.borrow_mut()[offset as usize] = value.clone()
        }
        fn read(&self, offset: u64, value: &mut String) {
            *value = self.borrow()[offset as usize].clone()
        }
    }

    impl Merkleable for String {
        fn zeros(i: usize) -> String {
            let mut s = "".to_string();
            for _ in 0..i {
                s.push(',')
            }
            s
        }
        fn hash(&self, another: &String) -> String {
            format!("{},{}", self, another)
        }
    }

    #[test]
    fn test_empty_tree() {
        let max_size = 32;
        let store: RefCell<Vec<String>> =
            RefCell::new((0..max_size).map(|_| String::new()).collect());
        let tree = FlatTree::new(&store, 0);
        assert!(tree.iter().next().is_none());
        assert!(tree.local_roots().is_empty());
        assert!(tree.full_roots().is_empty());
        assert!(tree.root().is_empty());
        assert!(tree.local_path(3).next().is_none());
        assert!(tree.full_roots().is_empty());
    }

    #[test]
    fn test_tree_depth() {
        let max_size = 32;
        let store: RefCell<Vec<String>> =
            RefCell::new((0..max_size).map(|_| String::new()).collect());
        let mut tree = FlatTree::new(&store, 0);
        assert_eq!(tree.depth(), 0);
        tree.push("0".to_string());
        assert_eq!(tree.depth(), 1);
        tree.push("1".to_string());
        assert_eq!(tree.depth(), 2);
        tree.push("2".to_string());
        assert_eq!(tree.depth(), 3);
        tree.push("3".to_string());
        assert_eq!(tree.depth(), 3);
        tree.push("4".to_string());
        assert_eq!(tree.depth(), 4);
    }

    #[test]
    fn test_tree() {
        assert_eq!(String::zeros(0), "");
        assert_eq!(String::zeros(1), ",");
        let start = |x: &str| Pos::Start(x.to_string());
        let left = |x: &str| Pos::Left(x.to_string());
        let right = |x: &str| Pos::Right(x.to_string());
        let max_size = 32;
        let store: RefCell<Vec<String>> =
            RefCell::new((0..max_size).map(|_| "X".to_string()).collect());
        let mut tree = FlatTree::new(&store, 0);
        for i in 0..3 {
            println!("++++++++++++++++ {}", i);
            tree.push(format!("{}", i));
            debug_print(&tree);
        }
        assert_eq!(
            tree.full_path(2).collect::<Vec<_>>(),
            [start("2"), right(""), left("0,1"),]
        );
        assert_eq!(
            tree.full_path_from_node_index(3).collect::<Vec<_>>(),
            [start("0,1,2,"),]
        );

        for i in 3..13 {
            println!("++++++++++++++++ {}", i);
            tree.push(format!("{}", i));
            debug_print(&tree);
        }
        assert_eq!(tree.depth(), 5);
        for (i, leaf) in tree.iter().enumerate() {
            assert_eq!(leaf, format!("{}", i));
        }
        assert_eq!(tree.local_roots(), ["0,1,2,3,4,5,6,7", "8,9,10,11", "12"]);
        assert_eq!(
            tree.local_path(3).collect::<Vec<_>>(),
            [start("3"), left("2"), left("0,1"), right("4,5,6,7")]
        );
        assert_eq!(
            tree.local_path(10).collect::<Vec<_>>(),
            [start("10"), right("11"), left("8,9")]
        );
        assert_eq!(tree.local_path(12).collect::<Vec<_>>(), [start("12")]);
        assert_eq!(
            tree.full_roots(),
            [
                (Index(15), "0,1,2,3,4,5,6,7,8,9,10,11,12,,,".to_string()),
                (Index(23), "8,9,10,11,12,,,".to_string()),
                (Index(27), "12,,,".to_string()),
                (Index(25), "12,".to_string()),
                (Index(24), "12".to_string()),
            ]
        );
        assert_eq!(
            tree.full_path(2).collect::<Vec<_>>(),
            [
                start("2"),
                right("3"),
                left("0,1"),
                right("4,5,6,7"),
                right("8,9,10,11,12,,,"),
            ]
        );
        assert_eq!(
            tree.full_path(3).collect::<Vec<_>>(),
            [
                start("3"),
                left("2"),
                left("0,1"),
                right("4,5,6,7"),
                right("8,9,10,11,12,,,"),
            ]
        );
        assert_eq!(
            tree.full_path(10).collect::<Vec<_>>(),
            [
                start("10"),
                right("11"),
                left("8,9"),
                right("12,,,"),
                left("0,1,2,3,4,5,6,7"),
            ]
        );
        assert_eq!(
            tree.full_path(12).collect::<Vec<_>>(),
            [
                start("12"),
                right(""),
                right(","),
                left("8,9,10,11"),
                left("0,1,2,3,4,5,6,7"),
            ]
        );
        assert_eq!(
            tree.full_path_from_node_index(3).collect::<Vec<_>>(),
            [start("0,1,2,3"), right("4,5,6,7"), right("8,9,10,11,12,,,"),]
        );
        assert_eq!(
            tree.full_path_from_node_index(9).collect::<Vec<_>>(),
            [
                start("4,5"),
                right("6,7"),
                left("0,1,2,3"),
                right("8,9,10,11,12,,,"),
            ]
        );
        assert_eq!(
            tree.full_path_from_node_index(15).collect::<Vec<_>>(),
            [start("0,1,2,3,4,5,6,7,8,9,10,11,12,,,"),]
        );
        assert_eq!(
            tree.full_path_from_node_index(23).collect::<Vec<_>>(),
            [start("8,9,10,11,12,,,"), left("0,1,2,3,4,5,6,7"),]
        );
        assert_eq!(
            tree.full_path_from_node_index(24).collect::<Vec<_>>(),
            [
                start("12"),
                right(""),
                right(","),
                left("8,9,10,11"),
                left("0,1,2,3,4,5,6,7")
            ]
        );
        assert_eq!(tree.full_path_from_node_index(26).collect::<Vec<_>>(), []);
        assert_eq!(
            tree.full_path_from_node_index(27).collect::<Vec<_>>(),
            [start("12,,,"), left("8,9,10,11"), left("0,1,2,3,4,5,6,7")],
        );
    }

    use proptest::prelude::*;

    proptest! {
      #[test]
      fn test_merkle_path(n in 0..30, i in 0..30) {
        let max_size = 64;
        if i < n {
            let store: RefCell<Vec<String>> =
                RefCell::new((0..max_size).map(|_| String::new()).collect());
            let mut tree = FlatTree::new(&store, 0);
            for x in 0..(n as usize) {
                tree.push(format!("{}", x))
            };
            let mut root = String::new();
            for p in tree.full_path(i as u64) {
                println!("p = {:?}", p);
                match p {
                    Pos::Start(x) => root = x,
                    Pos::Left(x) => root = x.hash(&root),
                    Pos::Right(x) => root = root.hash(&x),
                }
            }
            prop_assert_eq!(tree.root(), root);
        }
      }
    }
}
