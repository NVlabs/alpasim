---
name: doc-reviewer
description: Reviews recent code changes and checks if documentation needs updates. Reads all MD files in root and docs/ to identify stale or missing documentation. Use when completing features, before pushing, or when asked to "check docs", "review documentation", or "doc-reviewer".
---

# Documentation Reviewer

Reviews code changes since branching from main and checks all documentation for staleness.

## Quick Start

1. Get the list of changed files since branching from main
2. Discover and read all documentation files
3. Analyze which docs are affected by the code changes
4. Report findings with specific recommendations

## Workflow

### Step 1: Identify Changed Files

```bash
# Find the merge base with main
MERGE_BASE=$(git merge-base HEAD main 2>/dev/null || git merge-base HEAD origin/main)

# Get changed files with stats
git diff --stat $MERGE_BASE HEAD

# Get detailed list
git diff --name-only $MERGE_BASE HEAD
```

### Step 2: Discover and Read All Documentation

Find all markdown files in root and docs/:

```bash
# Discover all doc files
ls *.md docs/*.md 2>/dev/null
```

Read **all** discovered MD files. The total is typically ~40-50k tokens, well within context limits.

### Step 3: Analyze Impact

For each code change, check all documentation for:

1. **Contradictions** - Docs describe behavior differently than code now works
2. **Missing features** - New functionality not documented anywhere
3. **Stale references** - Removed code still mentioned in docs
4. **Outdated examples** - Code samples that no longer work
5. **Missing CLI options** - New flags or commands not documented
6. **Environment variables** - New or changed env vars
7. **Changed paths** - File/module paths that moved
8. **Dependency changes** - New packages or version requirements
9. **Build/test commands** - Changed invocation patterns

### Step 4: Report Findings

```markdown
## Documentation Review Summary

### Changes Analyzed
- X files changed across Y modules
- Commits: [describe range] since branching from main

### Documentation Status

#### Needs Update (High Priority)
- **[doc-path]**: [specific issue]
  - Code change: [what changed]
  - Doc gap: [what's missing or outdated]
  - Suggested fix: [specific text to add/change]

#### May Need Update (Review Recommended)
- **[doc-path]**: [potential issue]

#### Verified Current
- [List docs that were checked and are still accurate]

### Recommended Actions
1. [Ordered list of suggested documentation updates]
```

## What Triggers Documentation Updates

**High-priority (almost always needs doc update):**
- New public API functions or classes
- Changed function signatures or parameters
- New CLI commands or options
- New environment variables
- Changed installation or setup steps
- Modified build or test commands
- New dependencies
- Changed file/directory structure
- New services or modules

**Medium-priority (often needs doc update):**
- Internal refactoring that changes module organization
- Performance improvements worth noting
- Bug fixes that clarify expected behavior
- Configuration option changes
- New error messages or handling

**Low-priority (usually no doc update needed):**
- Internal implementation details unchanged externally
- Test-only changes
- Docstring improvements (self-documenting)
- Style/formatting changes
- Comments

## Example Session

```
User: Review the docs for my recent changes

Agent: Let me check what's changed since you branched from main.

[Runs git diff --stat against merge-base]

Found 8 files changed:
- src/runtime/dispatcher.py (new --parallel-workers option)
- src/wizard/cli.py (added --dry-run flag)
- pyproject.toml (bumped numpy version)

Discovering documentation files...

[Lists *.md and docs/*.md, then reads all of them]

## Documentation Review Summary

### Changes Analyzed
- 8 files changed across runtime/, wizard/, and root config
- Branch: feature/parallel-workers

### Needs Update (High Priority)

1. **docs/RUNNING_LOCALLY.md**: Missing --dry-run flag
   - Code: wizard/cli.py line 45 adds --dry-run option
   - Gap: CLI options section doesn't mention this flag
   - Fix: Add to "Wizard CLI Options" table:
     `| --dry-run | Validate config without executing |`

2. **AGENTS.md**: Outdated numpy reference
   - Code: pyproject.toml now requires numpy>=2.0
   - Gap: Build section mentions numpy 1.x compatibility
   - Fix: Update dependency note or remove version mention

### Verified Current
- docs/RUNTIME_PARALLELISM.md - Already covers worker parallelism
- docs/CODEBASE_MAP.md - Architecture unchanged
- CONTRIBUTING.md - Dev workflow unchanged
```

## Integration Points

**Before creating a PR:**
- Run doc-reviewer to catch missing documentation
- Include doc updates in the same PR as code

**During code review:**
- Reviewers can request doc-reviewer check
- Ensures docs stay synchronized

**Periodic maintenance:**
- Run against main branch for overall health check
- Compare `docs/CODEBASE_MAP.md` last_mapped date against recent commits
