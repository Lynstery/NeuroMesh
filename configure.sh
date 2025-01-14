#!/bin/bash
dir_input=$0
dir_input="${dir_input%/*}"
cd $dir_input
error=1

SERVER=0
if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_CONNECTION" ]; then
    SERVER=1
fi

shell=$(env | grep SHELL=)
echo $shell
if [ $shell == "SHELL=/bin/bash" ]; then
    rc=.bashrc
else
    rc=.zshrc
fi
NeuroMesh=$(pwd)
parent=$(dirname $NeuroMesh)
# gedit ${HOME}/.bashrc
# gedit ${HOME}/.zshrc
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "1" ]; then
        error=0
        echo "Configure NeuroMesh"
        if [ $(grep -c "# $NeuroMesh" ${HOME}/$rc) -eq 0 ]; then
            sed -i "\$a # $NeuroMesh" ${HOME}/$rc
            if [ $(grep -c "export PYTHONPATH=\${PYTHONPATH}:$parent$" ${HOME}/$rc) -eq 0 ]; then
                sed -i "\$a export PYTHONPATH=\${PYTHONPATH}:$parent" ${HOME}/$rc
            fi
        fi
        if [ "$SERVER" == "0" ]; then
            if [ $shell == "SHELL=/bin/bash" ]; then
                gnome-terminal -- /bin/sh -c 'source ${HOME}/$rc;exit'
            else
                gnome-terminal -- /bin/zsh -c 'source ${HOME}/$rc;exit'
            fi
        else
            echo "Please run source ${HOME}/$rc manually"
        fi
        echo "echo \${PYTHONPATH}"
    elif [ $1 == "0" ]; then
        error=0
        echo "Delete configure"
        NeuroMesh=$(echo "${NeuroMesh##*/}")
        sed -i "/$NeuroMesh/d" ${HOME}/$rc
        sed -i "\,export PYTHONPATH=\${PYTHONPATH}:$parent$,d" ${HOME}/$rc
        if [ "$SERVER" == "0" ]; then
            if [ $shell == "SHELL=/bin/bash" ]; then
                gnome-terminal -- /bin/sh -c 'source ${HOME}/$rc;exit'
            else
                gnome-terminal -- /bin/zsh -c 'source ${HOME}/$rc;exit'
            fi
        else
            echo "Please run source ${HOME}/$rc manually"
        fi
        echo "echo \${PYTHONPATH}"
    fi
fi
if [ $error == "1" ]; then
    echo "./configure.sh 1	Configure NeuroMesh"
    echo "./configure.sh 0	Delete configure"
fi
