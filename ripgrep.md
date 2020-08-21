# find files by name
- `rg -g 'pattern' --files `

# search in gzip, bzip2, xz, Brotli..
- `rg -z NEEDLE`

# restrict file type (see `rg --type-list`)
- `rg NEEDLE -tmd`
- `rg NEEDLE -truby`
- `rg NEEDLE -tpy`

