# Release Strategy and Roadmap

This document defines Shrimply's maturity levels, acceptance criteria for each release stage, and the roadmap from pre-alpha to 1.0.

## Release Maturity Levels

### Level 0: Pre-Alpha (Current)

**Current Status:** Version 0.0.x (development builds)

**Characteristics:**
- Unstable API and project file formats (breaking changes without notice)
- Frequent crashes, memory leaks, and resource issues
- Performance regressions common
- Project file breakage between versions (data loss possible)
- No security validation (plugins/scripts execute without sandboxing)
- Not recommended for production use

**What Works:**
- ✅ Basic video import/export
- ✅ Timeline editing (cuts, transitions, basic effects)
- ✅ GPU rendering (CUDA, wgpu paths)
- ✅ Cross-platform builds (Linux/Flatpak, partial macOS/Windows)

**Known Issues:**
- ❌ Project file stability
- ❌ Plugin security (arbitrary code execution)
- ❌ Performance optimization (memory usage, GPU utilization)
- ❌ macOS/Windows support incomplete
- ❌ Accessibility (AT-SPI coverage)
- ❌ Documentation (not user-facing yet)

**User SLA:** None. Pre-alpha is experimental; no support guarantee. Expect data loss.

**Lifespan:** Ongoing until Alpha acceptance criteria are met.

### Level 1: Alpha (Planned)

**Target Version:** 0.1.0  
**Target Date:** Q2-Q3 2027 (provisional; gates must pass)

**Characteristics:**
- Stable project file format (backwards-compatible for alpha versions)
- Major features complete (editing, effects, export)
- Reduced crash frequency (p95 uptime > 80% in typical workflows)
- Plugin sandboxing implemented (basic security)
- macOS and Windows builds functional (not fully optimized)
- User-facing documentation exists

**Acceptance Criteria Required:**
- ✅ Project files persist across alpha versions (forward-compatible within 0.1.x)
- ✅ Crash-free editing session (1 hour of typical workflows without crash)
- ✅ All major effects implemented and tested
- ✅ Plugin execution sandboxed (no unrestricted code execution)
- ✅ macOS and Windows builds pass CI and manual testing
- ✅ User guide and feature documentation (docs/source/user-guide/)
- ✅ Automated test coverage (integration tests for critical paths)
- ✅ Performance benchmarks recorded (baseline for regression detection)

**Known Gaps (Documented as Future Work):**
- ⚠️ Performance optimization (p95 GPU utilization measured, not optimized)
- ❌ Advanced color grading (tracked separately)
- ❌ Audio timeline (planned for Beta)
- ❌ Multi-threaded timeline (architecture planned, not implemented)

**User SLA:** Best-effort bug fixes within 2 weeks; crashes/data loss within 24 hours.

**Breaking Changes:** Project file format changes announced 2 weeks before release; migration script provided.

**Lifespan:** ~6 months or until Beta acceptance criteria are met.

### Level 2: Beta (Planned)

**Target Version:** 0.2.0  
**Target Date:** Q4 2027 (provisional)

**Characteristics:**
- Project file format stable (backwards-compatible for 1+ years)
- All planned features for 1.0 implemented and tested
- Uptime > 95% in typical workflows (p95 crash-free session > 4 hours)
- Performance optimized (GPU/CPU utilization targets met)
- Full macOS and Windows support (official testing matrix)
- Accessibility complete (AT-SPI compliance)
- Community translation infrastructure working

**Acceptance Criteria Required:**
- ✅ All Alpha criteria met
- ✅ Project file migration tools (beta → 1.0 guaranteed safe)
- ✅ Audio timeline complete and tested
- ✅ All planned effects/filters implemented
- ✅ Performance targets met (GPU utilization > 80%, memory < 2GB typical)
- ✅ Accessibility audit passed (keyboard navigation, screen reader support)
- ✅ macOS and Windows fully tested (official support matrix published)
- ✅ Community translation pipeline (i18n setup, language packs)
- ✅ Security audit completed (plugin sandbox, no known vulnerabilities)

**User SLA:** Bug fixes within 1 week; security/data-loss within 24 hours.

**Upgrade Path:** Beta users can upgrade within beta; migration to 1.0 guaranteed safe.

**Lifespan:** ~3 months or until 1.0 release.

### Level 3: General Availability (1.0)

**Target Version:** 1.0.0  
**Target Date:** Q1 2028 (provisional)

**Characteristics:**
- Production-ready software
- Project file format stable (backwards-compatible for 2+ years)
- Uptime > 99% (p95 crash-free session > 8 hours typical)
- Performance production-grade
- Full platform support (Linux, macOS, Windows all tier 1)
- Professional documentation and tutorials
- Support infrastructure in place

**Acceptance Criteria Required:**
- ✅ All Beta criteria met
- ✅ Performance benchmarks vs. industry peers (DaVinci Resolve, Premiere)
- ✅ Security audit by external firm (plugin sandbox, no critical issues)
- ✅ 99% uptime SLA-backed (monitored, incidents tracked)
- ✅ Professional user documentation (tutorials, workflows, troubleshooting)
- ✅ Keyboard shortcut customization (accessibility + power users)
- ✅ Plugin ecosystem mature (community plugins tested and curated)

**User SLA:**
- Bug fixes: 1-week SLA
- Security fixes: 24-hour SLA
- Critical data loss: incident response within 4 hours
- Maintenance window: Announced 1 month prior

**Support Channels:**
- GitHub issues (primary)
- Discourse forum or Discord (community)
- Email: support@shrimply.dev (when available)

**Upgrade Path:** Forward-compatible for 2 major versions; LTS versions supported for 3 years.

**Lifespan:** Indefinite; LTS until successor replaces it.

## Subsystem Maturity Matrix

| Subsystem | Pre-Alpha | Alpha | Beta | 1.0 | Notes |
|-----------|-----------|-------|------|-----|-------|
| **Core** | | | | | |
| Timeline editing | ✅ | ✅ | ✅ | ✅ | Fully implemented |
| Project persistence | ⚠️ | ✅ | ✅ | ✅ | Pre-alpha: data loss possible |
| Undo/redo | ✅ | ✅ | ✅ | ✅ | Functional |
| **Rendering** | | | | | |
| GPU (CUDA) | ✅ | ✅ | ✅ | ✅ | Primary path |
| GPU (wgpu/Metal) | ⚠️ | ✅ | ✅ | ✅ | Alpha: macOS support |
| CPU fallback | ⚠️ | ⚠️ | ✅ | ✅ | Beta: optimized |
| Real-time preview | ⚠️ | ✅ | ✅ | ✅ | Alpha: basic only |
| **Media** | | | | | |
| Video import | ✅ | ✅ | ✅ | ✅ | All common formats |
| Audio track | ❌ | ⚠️ | ✅ | ✅ | Alpha: basic, Beta: full |
| Image sequences | ⚠️ | ✅ | ✅ | ✅ | Pre-alpha: limited |
| Media library | ❌ | ❌ | ⚠️ | ✅ | Beta: basic, 1.0: full |
| **Effects** | | | | | |
| Color correction | ✅ | ✅ | ✅ | ✅ | Basic grading |
| Transitions | ✅ | ✅ | ✅ | ✅ | Cross-fades, wipes |
| Filters | ✅ | ✅ | ✅ | ✅ | Blur, sharpen, etc. |
| Motion graphics | ⚠️ | ✅ | ✅ | ✅ | Pre-alpha: basic |
| VFX (Manim) | ✅ | ✅ | ✅ | ✅ | Via plugins |
| **Plugins** | | | | | |
| Plugin system | ✅ | ✅ | ✅ | ✅ | Functional |
| Security sandbox | ❌ | ✅ | ✅ | ✅ | Alpha: basic |
| Plugin discovery | ❌ | ❌ | ⚠️ | ✅ | Beta: basic, 1.0: community |
| **Platforms** | | | | | |
| Linux/Flatpak | ✅ | ✅ | ✅ | ✅ | Primary platform |
| macOS | ⚠️ | ⚠️ | ✅ | ✅ | Pre/Alpha: limited |
| Windows | ⚠️ | ⚠️ | ✅ | ✅ | Pre/Alpha: limited |
| **UX** | | | | | |
| Keyboard shortcuts | ✅ | ✅ | ✅ | ✅ | Fixed set |
| Customizable shortcuts | ❌ | ❌ | ❌ | ✅ | 1.0 feature |
| Accessibility (a11y) | ❌ | ⚠️ | ✅ | ✅ | Alpha: partial |
| Dark mode | ✅ | ✅ | ✅ | ✅ | GTK4 default |
| **Documentation** | | | | | |
| Developer guide | ✅ | ✅ | ✅ | ✅ | In CONTRIBUTING.md |
| User guide | ❌ | ✅ | ✅ | ✅ | Alpha: online docs |
| Tutorials | ❌ | ⚠️ | ✅ | ✅ | Alpha: basic |
| Troubleshooting | ❌ | ⚠️ | ✅ | ✅ | Alpha: basic |

**Legend:**
- ✅ = Fully implemented and tested
- ⚠️ = Partially implemented or limited functionality
- ❌ = Not implemented

## High-Level Roadmap

```
NOW                      Q2-Q3 2027              Q4 2027            Q1 2028      Beyond
|                        |                       |                  |            |
Pre-Alpha 0.0.x          Alpha 0.1.0             Beta 0.2.0         1.0.0       LTS
├─ Timeline ✓            ├─ Stable project file  ├─ Full audio      ├─ 99% SLA  ├─ Security updates
├─ GPU rendering ✓       ├─ macOS/Windows       ├─ Optimization    ├─ Perf     ├─ i18n expansion
├─ Basic effects ✓       ├─ Documentation       ├─ Accessibility   │ targets   ├─ Community plugins
├─ Plugin system ✓       ├─ Tests/benchmarks    ├─ i18n pipeline   └─ Public   └─ LTS support
└─ Flatpak builds        └─ Plugin sandbox      └─ Security audit     release
```

## Contributor Priorities by Maturity Level

### Pre-Alpha → Alpha

**Critical Path:**
1. **Project file stability** — Implement versioning/migration for breaking changes
2. **macOS/Windows support** — Complete platform builds (CI, testing matrix)
3. **Plugin sandbox** — Basic security isolation (prevent arbitrary code execution)
4. **Crash stability** — Reduce crash frequency to < 20% of workflows
5. **Documentation** — User guide and troubleshooting

**Nice-to-Have:**
- Performance optimization
- Advanced color grading
- Community plugin examples

### Alpha → Beta

**Critical Path:**
1. **Audio timeline** — Full audio editing and mixing
2. **Performance** — Optimize GPU/CPU utilization to production targets
3. **Accessibility** — AT-SPI compliance, keyboard navigation
4. **Project file migration** — Guarantee safe upgrades to 1.0
5. **Security audit** — External review of plugin sandbox

**Nice-to-Have:**
- Advanced effects
- Media library
- Community translations

### Beta → 1.0

**Critical Path:**
1. **Keyboard customization** — Allow remapping shortcuts
2. **Professional docs** — Tutorials, workflows, API reference
3. **Plugin discovery** — Curated community plugin registry
4. **Performance benchmarks** — Published comparison vs. industry peers
5. **Support infrastructure** — Support email, forums, or Discord

**Nice-to-Have:**
- Advanced color grading
- Multi-threaded timeline
- Native plugins (Rust-based, lower overhead)

## Known Limitations (Pre-Alpha)

### Stability

- **Project files:** Not backwards-compatible between versions; data loss possible
- **Crashes:** Expected during complex operations (compositing, long timelines)
- **Memory:** No bounds on memory usage; unbounded growth possible
- **GPU:** CUDA-only in pre-alpha; wgpu/Metal paths incomplete

### Features

- **Plugin security:** No sandboxing; plugins run with full app privileges
- **Audio:** No audio timeline in pre-alpha; audio tracks read-only
- **Color grading:** Basic color wheels; advanced grading tools missing
- **Media library:** No asset organization or project templates

### Platform Support

- **macOS:** Partial support; performance not optimized
- **Windows:** Partial support; some dependencies missing
- **Linux:** Full support via Flatpak; native builds possible but unsupported

## Questions?

- Architecture questions: See CONTRIBUTING.md and docs/source/
- Feature questions: File an issue or discuss on GitHub
- Release timeline questions: Open an issue and label it `roadmap`
- Build/platform questions: See packaging/ directory
