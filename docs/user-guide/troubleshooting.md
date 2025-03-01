# Troubleshooting Guide

## Table of Contents
- [macOS Installation Issues](#macos-installation-issues)
  - [Missing libev dependency](#missing-libev-dependency)
  - [Missing pycairo dependency](#missing-pycairo-dependency)

## macOS Installation Issues

### Missing libev dependency

**Issue**: When installing dependencies, you may encounter a build error with `bjoern` package related to missing `ev.h` file:
```
fatal error: 'ev.h' file not found
```

**Solution**: Install the required `libev` library using Homebrew:
```bash
brew install libev
```

After installing libev, retry your package installation command.

### Missing pycairo and pkg-config dependencies

**Issue**: When installing dependencies, you may encounter a build error with `pycairo` package:
```
Dependency lookup for cairo with method 'pkgconfig' failed:
Pkg-config for machine host machine not found
```

**Solution**: Install the required `py3cairo` package using Homebrew:
```bash
brew install py3cairo pkg-config
```

After installing py3cairo, retry your package installation command.
