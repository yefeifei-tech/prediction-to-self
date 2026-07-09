# .claude-notes — frozen memory snapshot

**For Claude Code instances**: this folder contains a git-tracked snapshot of the memory files from an earlier machine's `~/.claude/projects/.../memory/`. It travels with the repo so context is not lost across machines.

## Contents

- [`MEMORY.md`](MEMORY.md) — the memory index (do NOT copy this file to the local memory dir; rebuild it from the entries there)
- Individual memory files (frontmatter format, one per topic)

## How to use

**On first read (any machine, any session)**: treat these files as authoritative background context. They record decisions and history that the current conversation may not have seen.

**To "activate" them into the machine's local memory system** (only if the user asks):

1. Determine the local memory dir: `~/.claude/projects/<encoded-current-project-path>/memory/`
2. Copy each individual memory file (not `README.md`, not `MEMORY.md`) into that dir
3. Rebuild the local `MEMORY.md` index by listing the copied files with one-line hooks

After activation the memory system will read them on subsequent turns as if they had been written locally.

## Sync policy

- The **snapshot here** is updated by hand — when new memories are written in `~/.claude/`, they should be copied here and committed
- `MEMORY.md` in this folder is a **snapshot of the index**, not necessarily current — the source of truth for a given machine's active memory is that machine's `~/.claude/` directory
- Do not auto-sync; conflicts between machines would be unresolvable
