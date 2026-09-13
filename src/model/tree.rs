use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use crate::scan::getattrlistbulk::{self, DirEntry};

/// Index path from root to a node in the tree (e.g. [2, 0, 1] = root's 3rd child → 1st child → 2nd child).
pub type TreePath = Vec<usize>;

thread_local! {
    static LOCAL_EXT_MAP: RefCell<HashMap<Box<str>, u64>> = RefCell::new(HashMap::new());
}

fn raw_extension(name: &str) -> &str {
    match name.rsplit_once('.') {
        Some((_, ext)) if !ext.is_empty() => ext,
        _ => "",
    }
}

/// A node in the file tree. Uses compact representation (Box<str> + Box<[T]>)
/// as validated by memory benchmarks: 40 bytes/struct, ~78 bytes RSS/node.
pub struct FileNode {
    pub name: Box<str>,
    pub size: u64,
    pub is_dir: bool,
    pub children: Box<[FileNode]>,
    /// Treemap rectangle, set during layout.
    pub rect: treemap::Rect,
    /// Cached file count (1 for files, sum of children for dirs).
    pub file_count: u64,
    /// Cached directory count (0 for files, 1 + sum of children for dirs).
    pub dir_count: u64,
}

impl FileNode {
    /// Get the file extension, or empty string for dirs/extensionless files.
    pub fn extension(&self) -> &str {
        if self.is_dir {
            ""
        } else {
            raw_extension(&self.name)
        }
    }

    /// Walk a path of child indices to reach a descendant node.
    pub fn resolve_path(&self, path: &[usize]) -> Option<&FileNode> {
        let mut node = self;
        for &idx in path {
            node = node.children.get(idx)?;
        }
        Some(node)
    }

    /// Remove the child at `index` from this node's children, updating size and counts.
    /// Returns the removed child.
    pub fn remove_child(&mut self, index: usize) -> FileNode {
        let mut children = std::mem::take(&mut self.children).into_vec();
        let removed = children.remove(index);
        self.size = self.size.saturating_sub(removed.size);
        self.file_count = self.file_count.saturating_sub(removed.file_count);
        self.dir_count = self.dir_count.saturating_sub(removed.dir_count);
        self.children = children.into();
        removed
    }
}

/// The complete scanned file tree with precomputed extension statistics.
pub struct FileTree {
    pub root: FileNode,
    pub root_path: String,
    /// Extension -> total bytes mapping, sorted by size descending.
    pub extensions: Vec<(Box<str>, u64)>,
}

impl FileTree {
    /// Build a file tree by scanning the given path using getattrlistbulk.
    pub fn scan(root: &Path) -> Self {
        let ext_map = Mutex::new(HashMap::<Box<str>, u64>::new());
        let root_node = build_root_node(root);

        // Drain the main thread's local ext map
        LOCAL_EXT_MAP.with(|m| {
            let local = m.replace(HashMap::new());
            if !local.is_empty() {
                let mut global = ext_map.lock().unwrap_or_else(|e| e.into_inner());
                for (k, v) in local {
                    *global.entry(k).or_default() += v;
                }
            }
        });

        // Drain all rayon worker thread local ext maps
        rayon::broadcast(|_| {
            LOCAL_EXT_MAP.with(|m| {
                let local = m.replace(HashMap::new());
                if !local.is_empty() {
                    let mut global = ext_map.lock().unwrap_or_else(|e| e.into_inner());
                    for (k, v) in local {
                        *global.entry(k).or_default() += v;
                    }
                }
            });
        });

        let mut extensions: Vec<(Box<str>, u64)> = ext_map
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .into_iter()
            .collect();
        extensions.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        FileTree {
            root: root_node,
            root_path: root.display().to_string(),
            extensions,
        }
    }

    /// Build the full filesystem path for a node identified by index path.
    pub fn build_fs_path(&self, path: &[usize]) -> Option<std::path::PathBuf> {
        let mut fs_path = std::path::PathBuf::from(&self.root_path);
        let mut node = &self.root;
        for &idx in path {
            let child = node.children.get(idx)?;
            fs_path.push(&*child.name);
            node = child;
        }
        Some(fs_path)
    }

    /// Remove the node at the given index path from the tree, updating all ancestor sizes/counts.
    /// Returns the removed node, or None if the path is invalid.
    pub fn remove_at_path(&mut self, path: &[usize]) -> Option<FileNode> {
        let (&child_idx, parent_path) = path.split_last()?;

        // Navigate to the parent
        let mut node = &mut self.root;
        for &idx in parent_path {
            node = node.children.get_mut(idx)?;
        }

        if child_idx >= node.children.len() {
            return None;
        }

        Some(node.remove_child(child_idx))
    }

    /// Propagate a size/count reduction up the ancestor chain (excluding the node itself).
    pub fn subtract_from_ancestors(
        &mut self,
        path: &[usize],
        size: u64,
        file_count: u64,
        dir_count: u64,
    ) {
        // The direct parent is already updated by remove_child; update grandparents and above.
        let mut node = &mut self.root;
        // Update root
        node.size = node.size.saturating_sub(size);
        node.file_count = node.file_count.saturating_sub(file_count);
        node.dir_count = node.dir_count.saturating_sub(dir_count);
        // Update intermediate ancestors (not the direct parent, which remove_child handles)
        if path.len() >= 2 {
            for &idx in &path[..path.len() - 2] {
                if let Some(child) = node.children.get_mut(idx) {
                    child.size = child.size.saturating_sub(size);
                    child.file_count = child.file_count.saturating_sub(file_count);
                    child.dir_count = child.dir_count.saturating_sub(dir_count);
                    node = child;
                } else {
                    break;
                }
            }
        }
    }

    /// Rebuild extension statistics from the current tree.
    pub fn rebuild_extensions(&mut self) {
        let mut ext_map: HashMap<Box<str>, u64> = HashMap::new();
        collect_extensions(&self.root, &mut ext_map);
        let mut extensions: Vec<(Box<str>, u64)> = ext_map.into_iter().collect();
        extensions.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        self.extensions = extensions;
    }
}

fn collect_extensions(node: &FileNode, map: &mut HashMap<Box<str>, u64>) {
    if !node.is_dir {
        let ext = node.extension();
        if !ext.is_empty() {
            *map.entry(ext.into()).or_default() += node.size;
        } else {
            *map.entry("(no ext)".into()).or_default() += node.size;
        }
    }
    for child in node.children.iter() {
        collect_extensions(child, map);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    /// Scanning must not follow symlinks: a symlink to a dir stays a childless
    /// file node, its target is counted once, and a self-loop terminates.
    #[test]
    fn scan_does_not_follow_symlinks() {
        let base = std::env::temp_dir().join(format!(
            "macdirstat-symlink-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("real")).unwrap();
        std::fs::write(base.join("real/file.txt"), vec![b'x'; 1024]).unwrap();
        symlink("real", base.join("link_to_real")).unwrap();
        symlink("loop", base.join("loop")).unwrap();

        let tree = FileTree::scan(&base);

        let find = |name: &str| {
            tree.root
                .children
                .iter()
                .find(|c| &*c.name == name)
                .unwrap_or_else(|| panic!("missing child {name}"))
        };
        let link = find("link_to_real");
        assert!(!link.is_dir, "symlink to a dir must not be a dir node");
        assert!(link.children.is_empty());
        let lp = find("loop");
        assert!(!lp.is_dir, "symlink loop must not be a dir node");
        let real = find("real");
        assert!(real.is_dir);
        assert_eq!(real.children.len(), 1);
        assert!(
            link.size < real.size,
            "symlink must report link size, not target size"
        );
        let sum: u64 = tree.root.children.iter().map(|c| c.size).sum();
        assert_eq!(tree.root.size, sum, "target must be counted once");

        let _ = std::fs::remove_dir_all(&base);
    }

    /// Same directory opened twice (e.g. firmlink mirror paths share dev+ino):
    /// the second claim must be rejected so its contents aren't counted twice.
    #[test]
    fn claim_dir_rejects_second_fd_to_same_dir() {
        let base = std::env::temp_dir().join(format!(
            "macdirstat-visited-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();

        let visited = VisitedDirs::default();
        let fd1 = getattrlistbulk::open_dir(&base);
        assert!(fd1 >= 0);
        assert!(claim_ident(dir_ident(fd1), &visited));
        let fd2 = getattrlistbulk::open_dir(&base);
        assert!(fd2 >= 0);
        assert!(
            !claim_ident(dir_ident(fd2), &visited),
            "same dir via a second fd must be rejected"
        );
        getattrlistbulk::close_dir(fd1);
        getattrlistbulk::close_dir(fd2);

        let _ = std::fs::remove_dir_all(&base);
    }

    /// Distinct directories must never be pruned: no false-positive dedup.
    #[test]
    fn scan_keeps_distinct_directories() {
        let base = std::env::temp_dir().join(format!(
            "macdirstat-distinct-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("a")).unwrap();
        std::fs::create_dir_all(base.join("b")).unwrap();
        std::fs::write(base.join("a/file.txt"), vec![b'x'; 512]).unwrap();
        std::fs::write(base.join("b/file.txt"), vec![b'y'; 512]).unwrap();

        let tree = FileTree::scan(&base);

        let find = |name: &str| {
            tree.root
                .children
                .iter()
                .find(|c| &*c.name == name)
                .unwrap_or_else(|| panic!("missing child {name}"))
        };
        let a = find("a");
        let b = find("b");
        assert!(a.is_dir && b.is_dir);
        assert_eq!(a.children.len(), 1);
        assert_eq!(b.children.len(), 1);
        assert_eq!(tree.root.dir_count, 3, "root + a + b");

        let _ = std::fs::remove_dir_all(&base);
    }

    /// Firmlink mirror matching: exact targets and anything beneath them
    /// match; siblings and lookalike prefixes must not.
    #[test]
    fn firmlink_mirror_matching() {
        let targets: HashSet<Box<str>> =
            ["Users".into(), "System/Library/Caches".into(), "usr/local".into()]
                .into_iter()
                .collect();
        assert!(is_firmlink_mirror("Users", &targets));
        assert!(is_firmlink_mirror("Users/gregoreesmaa", &targets));
        assert!(is_firmlink_mirror("System/Library/Caches", &targets));
        assert!(!is_firmlink_mirror("System/Library", &targets));
        assert!(!is_firmlink_mirror("System", &targets));
        assert!(!is_firmlink_mirror("Users2", &targets));
        assert!(!is_firmlink_mirror("Applications", &targets));
        assert!(!is_firmlink_mirror("usr/libexec", &targets));
        assert!(is_firmlink_mirror("usr/local/bin", &targets));
    }

    /// End-to-end prune through real traversal: with a synthetic ctx treating
    /// the fixture root as the Data volume and `a` as a mirror target, `a`
    /// disappears while its sibling `b` scans normally.
    #[test]
    fn firmlink_mirror_subtree_is_skipped() {
        let base = std::env::temp_dir().join(format!(
            "macdirstat-mirror-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("a")).unwrap();
        std::fs::create_dir_all(base.join("b")).unwrap();
        std::fs::write(base.join("a/file.txt"), vec![b'x'; 512]).unwrap();
        std::fs::write(base.join("b/file.txt"), vec![b'y'; 512]).unwrap();

        let base_fd = getattrlistbulk::open_dir(&base);
        let base_id = dir_ident(base_fd);
        assert!(base_id.is_some());
        getattrlistbulk::close_dir(base_fd);
        let ctx = ScanCtx {
            visited: VisitedDirs::default(),
            firmlinks: ["a".into()].into_iter().collect(),
            data_root: base_id,
        };
        let root_fd = getattrlistbulk::open_dir(&base);
        // The fixture root plays the Data volume: its Data-relative path is "".
        let node = build_node_fd(root_fd, "base".into(), &ctx, Some(""));
        getattrlistbulk::close_dir(root_fd);

        assert_eq!(node.children.len(), 1);
        assert_eq!(&*node.children[0].name, "b");
        assert_eq!(node.children[0].children.len(), 1);

        let _ = std::fs::remove_dir_all(&base);
    }

    /// The real firmlink table parses to the expected targets (macOS only).
    #[test]
    fn firmlink_table_loads() {
        if !std::path::Path::new("/usr/share/firmlinks").exists() {
            return; // not macOS — nothing to parse
        }
        let targets = load_firmlink_targets();
        assert!(targets.contains("Users"));
        assert!(targets.contains("System/Library/Caches"));
        for t in targets.iter() {
            assert!(!t.is_empty());
            assert!(!t.starts_with('/'));
            assert!(!t.ends_with('/'));
        }
    }
}

/// Identity of a visited directory: (device, inode).
/// Firmlinks expose the same directory at two paths (e.g. `/Users` and
/// `/System/Volumes/Data/Users`) with identical dev+ino; without tracking,
/// a scan covering both views counts everything twice.
type VisitedDirs = Mutex<HashSet<(u64, u64)>>;

/// Per-scan dedup state shared across traversal threads.
struct ScanCtx {
    visited: VisitedDirs,
    /// Data-relative firmlink targets from `/usr/share/firmlinks`
    /// (e.g. `Users`, `System/Library/Caches`). Empty when the table is
    /// unreadable or the scan root lives inside the Data volume (single
    /// view — nothing to disambiguate), in which case only `visited` applies.
    firmlinks: HashSet<Box<str>>,
    /// (dev, ino) of `/System/Volumes/Data`, or None when disabled as above.
    data_root: Option<(u64, u64)>,
}

/// (dev, ino) identity of an open directory fd, or None if it can't be stated.
fn dir_ident(fd: libc::c_int) -> Option<(u64, u64)> {
    if fd < 0 {
        return None;
    }
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(fd, &mut st) } != 0 {
        return None;
    }
    Some((st.st_dev as u64, st.st_ino as u64))
}

/// Claim an identity for traversal. Returns false when already visited
/// through another path. Fails open (returns true) when unstated.
fn claim_ident(ident: Option<(u64, u64)>, visited: &VisitedDirs) -> bool {
    match ident {
        None => true,
        Some(key) => visited
            .lock()
            .map(|mut guard| guard.insert(key))
            .unwrap_or(true),
    }
}

/// Whether `path` lives inside the Data volume (single firmlink view —
/// the mirror table must not prune anything). String-prefix is enough:
/// a wrong answer only falls back to visited-set dedup, never to a wrong total.
fn path_inside_data(path: &Path) -> bool {
    path.starts_with("/System/Volumes/Data")
}

/// Load Data-relative firmlink targets (`Users`, `usr/local`, ...) from the
/// system firmlink table. Returns an empty set when unreadable.
fn load_firmlink_targets() -> HashSet<Box<str>> {
    let mut set = HashSet::new();
    let Ok(text) = std::fs::read_to_string("/usr/share/firmlinks") else {
        return set;
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((_, target)) = line.split_once('\t') {
            let target = target.trim().trim_matches('/');
            if !target.is_empty() {
                set.insert(target.into());
            }
        }
    }
    set
}

/// Whether the Data-relative path `rel` (e.g. `Users/gregoreesmaa`) is a
/// firmlink mirror: it equals a target or sits beneath one, so the same
/// directory is also reachable via its canonical path.
fn is_firmlink_mirror(rel: &str, targets: &HashSet<Box<str>>) -> bool {
    let mut prefix = rel;
    loop {
        if targets.contains(prefix) {
            return true;
        }
        match prefix.rsplit_once('/') {
            Some((parent, _)) => prefix = parent,
            None => return false,
        }
    }
}

fn build_root_node(path: &Path) -> FileNode {
    let fd = getattrlistbulk::open_dir(path);
    if fd < 0 {
        eprintln!(
            "Warning: could not open directory {:?} (permission denied or not found)",
            path
        );
    }
    let firmlinks = if path_inside_data(path) {
        HashSet::new()
    } else {
        load_firmlink_targets()
    };
    let data_root = if firmlinks.is_empty() {
        None
    } else {
        let data_fd = getattrlistbulk::open_dir(Path::new("/System/Volumes/Data"));
        let ident = dir_ident(data_fd);
        getattrlistbulk::close_dir(data_fd);
        ident
    };
    let ctx = ScanCtx {
        visited: VisitedDirs::default(),
        firmlinks,
        data_root,
    };
    let root_ident = dir_ident(fd);
    claim_ident(root_ident, &ctx.visited);
    // Data-relative path of the root itself ("" when root IS the Data volume).
    let root_rel = match root_ident {
        Some(id) if ctx.data_root == Some(id) => Some(String::new()),
        _ => None,
    };
    let name: Box<str> = path.display().to_string().into();
    let node = build_node_fd(fd, name, &ctx, root_rel.as_deref());
    getattrlistbulk::close_dir(fd);
    node
}

/// Build a FileNode from an already-opened directory fd.
/// `node_name` is the display name for this node.
/// `ctx` dedupes directories reachable via multiple paths (firmlinks);
/// `data_rel` is this directory's path relative to `/System/Volumes/Data`
/// (None when outside the Data volume — the common case, zero cost).
fn build_node_fd(
    parent_fd: libc::c_int,
    node_name: Box<str>,
    ctx: &ScanCtx,
    data_rel: Option<&str>,
) -> FileNode {
    use rayon::prelude::*;

    let entries = getattrlistbulk::scan_dir_entries_fd(parent_fd);

    // Separate files and directories
    let mut file_nodes: Vec<FileNode> = Vec::new();
    let mut dir_names: Vec<&DirEntry> = Vec::new();
    let mut total_size: u64 = 0;
    let mut total_file_count: u64 = 0;

    for entry in &entries {
        if entry.is_dir {
            dir_names.push(entry);
        } else {
            total_size += entry.file_size;
            total_file_count += 1;
            LOCAL_EXT_MAP.with(|m| {
                let mut map = m.borrow_mut();
                let ext = raw_extension(&entry.name);
                let key: Box<str> = if ext.is_empty() {
                    "(no ext)".into()
                } else {
                    ext.into()
                };
                *map.entry(key).or_default() += entry.file_size;
            });
            file_nodes.push(FileNode {
                name: entry.name.clone(),
                size: entry.file_size,
                is_dir: false,
                children: Box::new([]),
                rect: treemap::Rect::new(),
                file_count: 1,
                dir_count: 0,
            });
        }
    }

    // Recurse into subdirectories — use openat() relative to parent fd.
    // Two dedup layers: firmlink mirrors under /System/Volumes/Data are
    // always skipped (their canonical path covers them — deterministic),
    // and anything already visited via another path is skipped as well
    // (bind mounts and friends — first claim wins).
    let build_child = |entry: &&DirEntry| -> Option<FileNode> {
        let child_fd = getattrlistbulk::openat_dir(parent_fd, &entry.name);
        let ident = dir_ident(child_fd);
        // Track the Data-relative path once inside the Data volume. No
        // allocation happens outside it (the common case).
        let child_rel: Option<String> = match (ident, data_rel) {
            (Some(id), _) if ctx.data_root == Some(id) => Some(String::new()),
            (_, Some(parent)) if !ctx.firmlinks.is_empty() => Some(if parent.is_empty() {
                entry.name.to_string()
            } else {
                format!("{parent}/{}", &entry.name)
            }),
            _ => None,
        };
        if let Some(ref rel) = child_rel {
            if !rel.is_empty() && is_firmlink_mirror(rel, &ctx.firmlinks) {
                getattrlistbulk::close_dir(child_fd);
                return None;
            }
        }
        if !claim_ident(ident, &ctx.visited) {
            getattrlistbulk::close_dir(child_fd);
            return None;
        }
        let node = build_node_fd(child_fd, entry.name.clone(), ctx, child_rel.as_deref());
        getattrlistbulk::close_dir(child_fd);
        Some(node)
    };

    let dir_nodes: Vec<FileNode> = if dir_names.len() >= 2 {
        dir_names.par_iter().filter_map(build_child).collect()
    } else {
        dir_names.iter().filter_map(build_child).collect()
    };

    let mut total_dir_count: u64 = 0;
    for child in &dir_nodes {
        total_size += child.size;
        total_file_count += child.file_count;
        total_dir_count += child.dir_count;
    }

    let mut children: Vec<FileNode> = Vec::with_capacity(file_nodes.len() + dir_nodes.len());
    children.extend(file_nodes);
    children.extend(dir_nodes);

    children.sort_unstable_by(|a, b| b.size.cmp(&a.size));

    FileNode {
        name: node_name,
        size: total_size,
        is_dir: true,
        children: children.into(),
        rect: treemap::Rect::new(),
        file_count: total_file_count,
        dir_count: total_dir_count + 1,
    }
}
