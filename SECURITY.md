# Security policy

Magnelio is a numerical library: it reads model descriptions and CAD
files, runs a solver, and writes result files.  It opens no network
connections and executes no code from the files it reads.  The
findings that matter here are therefore of the kind "a crafted STEP,
Gerber, HDF5 or project file makes the library write outside the
project directory, or exhaust memory in a way that is not just a
large model".

## Reporting a vulnerability

Please do not open a public issue for a security-relevant finding.
Use one of the two private routes:

- GitHub's private vulnerability reporting:
  <https://github.com/brtkrtz/magnelio/security/advisories/new>
- e-mail: <magnelio@brtkrtz.de>

You will get an acknowledgement within a week.  Fixes ship as a patch
release with a changelog entry; the report is credited if you wish.

## Supported versions

Only the latest release on PyPI and conda-forge receives fixes.
