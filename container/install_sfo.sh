#!/bin/bash

# cd
# git clone https://github.com/dengwxn/ray.git Ray-SFO

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
	build-essential \
	cmake \
	git \
	wget \
	curl \
	zsh \
	vim \
	python3 \
	python3-pip \
	python3-dev \
	python3-setuptools \
	python3-venv
rm -rf /var/lib/apt/lists/*

sudo apt-get update
sudo apt-get install -y build-essential curl clang-12 pkg-config psmisc unzip

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

echo 'plugins=(git zsh-autosuggestions)' >>~/.zshrc
echo 'ZSH_THEME="robbyrussell"' >>~/.zshrc

cd ~/Ray-SFO
git checkout ddp-dev-link-0213
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel

~/Ray-SFO/ci/env/install-bazel.sh --user
export PATH="$PATH:$HOME/bin"
bazel --version

cd
wget https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh
chmod +x install.sh
./install.sh
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
rm install.sh
nvm install 14
nvm use 14

cd ~/Ray-SFO/python/ray/dashboard/client
npm ci
npm run build

cd ~/Ray-SFO/python
pip install -r requirements.txt
pip install -e . --verbose
pip install pytest torch
pip install -c requirements_compiled.txt -r requirements/lint-requirements.txt
pip install --upgrade typing-extensions

git config --global user.email "weixin@cs.washington.edu"
git config --global user.name "Weixin Deng"

# [TODO]
# 1. Output the `nvidia-smi` to file.
# 2. Optionally push the `results/sfo` to remote.
# 3. Optionally output the elapse time for each script.
