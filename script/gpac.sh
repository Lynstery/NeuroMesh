git clone https://github.com/gpac/gpac.git gpac_public
git clone https://github.com/gpac/deps_unix
cd deps_unix
git submodule update --init --recursive --force --checkout
./build_all.sh linux
cd ../gpac_public
./configure
make
sudo make install
rm -rf gpac_public 
rm -rf deps_unix