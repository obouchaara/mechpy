#!/bin/bash

# Exit if any command fails
set -e

# Build the docs
sphinx-build -M html docs/source/ docs/build/

# Stash any changes in main branch
git stash -u

# Switch to gh-pages branch
git checkout gh-pages

# Copy the HTML files to the root of gh-pages
cp -r docs/build/html/* .

# Add all changes to git
git add .

# Commit the changes
git commit -m "Update documentation"

# Push to the remote gh-pages branch
git push origin gh-pages

# Switch back to main branch
git checkout main

# Apply stashed changes
git stash pop
