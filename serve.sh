#!/usr/bin/env bash
# Local dev server for the al-folio site (deo-ahn design + your content).
# Usage: ./serve.sh           → http://127.0.0.1:4001/
#        ./serve.sh -P 4002   → pass extra jekyll flags
set -e
cd "$(dirname "$0")"

# Homebrew Ruby 4.x (system Ruby 2.6 is too old for al-folio).
export PATH="/usr/local/opt/ruby/bin:$PATH"

# REQUIRED: bibtex-ruby / jekyll-scholar fail with "invalid byte sequence in
# US-ASCII" on non-ASCII .bib content unless the locale is UTF-8.
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export RUBYOPT="-EUTF-8"

bundle exec jekyll serve --host 127.0.0.1 --port 4001 --livereload "$@"
