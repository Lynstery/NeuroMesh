sudo apt install automake autoconf libtool m4 build-essential
git clone https://github.com/ultravideo/kvazaar.git
cd kvazaar
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
cd ..
rm -rf kvazaar