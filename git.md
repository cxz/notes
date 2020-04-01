# sort branches by commit date
git branch --sort=-committerdate

# Checkout branch N number of checkouts ago
git checkout @{-N}

# Checkout previous branch; `-` acts as shorthand for `@{-1}`
git checkout -

# List branches along with commit ID, commit message and remote
git branch -vv

# checkout file from other branch
git checkout feature/my-other-branch -- thefile.txt

# shorter git status
git status -sb

# See the reference log of your activity
git reflog --all

# Look at the HEAD at given point from reflog
git show HEAD@{2}

# Checkout the HEAD, to get back to that point
git checkout HEAD@{2}

# This will find any change that was staged but is not attached to the git tree
git fsck --lost-found

# See the dates of the files
ls -lah .git/lost-found/other/

# Copy the relevant files to where you want them, for example:
cp .git/lost-found/other/73f60804ac20d5e417783a324517eba600976d30 index.html


