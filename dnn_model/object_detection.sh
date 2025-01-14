#!/bin/bash
torchvision=$(python -c "import torchvision; print(torchvision.__path__[0])")
faster_rcnn=${torchvision}/models/detection/generalized_rcnn.py
retinanet=${torchvision}/models/detection/retinanet.py
if [ ! -f "${faster_rcnn}" ]; then
    echo "${faster_rcnn}/ dose not exist"
    exit
fi
echo ${faster_rcnn}
if [ ! -f "${retinanet}" ]; then
    echo "${retinanet}/ dose not exist"
    exit
fi
echo ${retinanet}
error=1
if [ -n "$1" ] && [ ! -n "$2" ]; then
    if [ "$1" == "1" ]; then
        error=0
        sed -i "s/def forward(self, images, targets=None):/def forward(self, images, targets=None, disable_post=False):/g" ${faster_rcnn}
        if ! grep -q "if disable_post:" ${faster_rcnn}; then
            sed -i "s/if torch.jit.is_scripting():/if disable_post:\n            return (features, proposals)\n        if torch.jit.is_scripting():/g" ${faster_rcnn}
        fi
        sed -i "s/def forward(self, images, targets=None):/def forward(self, images, targets=None, disable_post=False):/g" ${retinanet}
        if ! grep -q "if disable_post:" ${retinanet}; then
            sed -i "s/if torch.jit.is_scripting():/if disable_post:\n            return head_outputs\n        if torch.jit.is_scripting():/g" ${retinanet}
        fi
    elif [ $1 == "0" ]; then
        error=0
        sed -i "s/def forward(self, images, targets=None, disable_post=False):/def forward(self, images, targets=None):/g" ${faster_rcnn}
        sed -i '/if disable_post:/,+1d' ${faster_rcnn}
        sed -i "s/def forward(self, images, targets=None, disable_post=False):/def forward(self, images, targets=None):/g" ${retinanet}
        sed -i '/if disable_post:/,+1d' ${retinanet}
    fi
fi
if [ $error == "1" ]; then
    echo "./object_detection.sh 1	Disable post-processing on inference results"
    echo "./object_detection.sh 0	Allow post-processing on inference results"
fi
