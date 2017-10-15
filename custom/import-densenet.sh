# importer.sh
# Import pretrained pytorch models
#
# --------------------------------------------------------
# mcnPyTorch
# Licensed under The MIT License [see LICENSE.md for details]
# Copyright (C) 2017 Samuel Albanie 
# --------------------------------------------------------

import_dir="models"
refresh_models=false
test_imported_models=false # (saves features to disk for comparison)

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

function convert_model()
{
    pytorch_model=$1
    mcn_model_path=$2
    model_def=$3
    weights=$4
	converter="ipython $SCRIPTPATH/../python/import_pytorch.py --"
	mkdir -p "${import_dir}"

    if [ $refresh_models = false ] && [ -e $mcn_model_path ]
	then
		echo "$mcn_model_path already exists; skipping."
	else echo "Exporting PyTorch model to matconvnet (may take some time) ..." 
        $converter \
                --image-size='[224,224]' \
                --full-image-size='[256,256]' \
                --model-def=$model_def \
                --model-weights=$weights \
                $pytorch_model $mcn_model_path
    fi

    if [ $test_imported_models = true ]
    then 
        tester="ipython $SCRIPTPATH/../test/dump_pytorch_features.py --"
        $tester \
                --image-size='[224,224]' \
                $pytorch_model $mcn_model_path
    fi
}
	
# Example models from the torchvision module
declare -a model_list=(
"densenet121" 
"densenet161" 
"densenet169" 
"densenet201" 
)

for pytorch_model in "${model_list[@]}"
do
    mcn_model_path="${import_dir}/${pytorch_model}-pt-mcn.mat"
    convert_model $pytorch_model $mcn_model_path
done
