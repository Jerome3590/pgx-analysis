#!/usr/bin/env bash
# AL2 emergency only. Do not use for new sessions.
# New work: Amazon Linux 2023 + current requirements.txt
# (aws-pgx-setup/ec2/README_pgx_session.md).
# On-box: finish Python 3.11 after an R/bupaverse abort. Does not compile R.
set -euxo pipefail
export PATH="/usr/local/openssl/bin:/usr/local/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/openssl/lib:${LD_LIBRARY_PATH:-}"
LOG="${LOG:-/var/log/resume_python_no_r.log}"
exec > >(tee -a "$LOG") 2>&1
echo "==== START $(date -u) ===="

if [[ ! -x /usr/local/bin/python3.11 ]]; then
  yum -y install sqlite-devel gdbm-devel libdb-devel libffi-devel \
    bzip2-devel xz-devel ncurses-devel readline-devel tk-devel
  cd /usr/local
  if [[ ! -d Python-3.11.9 ]]; then
    wget -q https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
    tar xzf Python-3.11.9.tgz
  fi
  cd Python-3.11.9
  ./configure --with-openssl=/usr/local/openssl --prefix=/usr/local --enable-shared
  make -j "$(nproc)"
  make altinstall
  echo '/usr/local/lib' > /etc/ld.so.conf.d/python3.11.conf
  ldconfig
fi
ln -sf /usr/local/bin/python3.11 /usr/bin/python3.11
ln -sf /usr/local/bin/python3.11 /usr/bin/python3
ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3 2>/dev/null || true

/usr/local/bin/python3.11 -m pip install --upgrade pip
# AL2 GCC 7.3 cannot build NumPy 2.x (needs GCC >= 9.3). Pin the bootstrap set.
/usr/local/bin/python3.11 -m pip install --only-binary=:all: 'numpy==1.26.4' 'pandas==2.2.3' pyarrow
# AL2 is glibc 2.26; DuckDB 1.4+ httpfs extensions need GLIBC_2.28.
/usr/local/bin/python3.11 -m pip install boto3 'duckdb==1.1.3'
/usr/local/bin/python3.11 -c "import duckdb,pandas,boto3,pyarrow; print('python-ready')"
echo "==== DONE $(date -u) ===="
