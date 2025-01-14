ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    # https://docs.anaconda.com/free/anaconda/install/linux/
    url="https://repo.anaconda.com/archive/"
    latest_version=$(curl -s "$url" | grep href | sed 's/.*href="//' | sed 's/".*//' | awk '/Linux-x86_64.sh/{print; exit}')
elif [ "$ARCH" = "aarch64" ]; then
    url="https://repo.anaconda.com/miniconda/"
    latest_version="Miniconda3-latest-Linux-aarch64.sh"
fi
if [ ! -f "$latest_version" ]; then
    curl -O $url$latest_version
fi
chmod +x $latest_version
echo "注意：面对\"Do you wish to update your shell profile to automatically initialize conda?\"时应选择yes"
./$latest_version
rm -rf $latest_version