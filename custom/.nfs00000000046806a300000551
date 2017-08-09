# Import the Inception v3 model 
#
# --------------------------------------------------------
# mcnPyTorch
# Licensed under The MIT License [see LICENSE.md for details]
# Copyright (C) 2017 Samuel Albanie 
# --------------------------------------------------------

import_dir="models"
refresh_models=false
test_imported_models=true # (saves features to disk for comparison)

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

function convert_model()
{
    pytorch_model=$1
    mcn_model_path=$2
    model_def=$3
    weights=$4
	converter="ipython --pdb $SCRIPTPATH/../python/import_pytorch.py --"
	mkdir -p "${import_dir}"

    if [ $refresh_models = false ] && [ -e $mcn_model_path ]
	then
		echo "$mcn_model_path already exists; skipping."
	else echo "Exporting PyTorch model to matconvnet (may take some time) ..." 
        $converter \
                --image-size='[299,299]' \
                --full-image-size='[299,299]' \
                --model-def=$model_def \
                --model-weights=$weights \
                $pytorch_model $mcn_model_path
    fi

    if [ $test_imported_models = true ]
    then 
        tester="ipython $SCRIPTPATH/../test/dump_pytorch_features.py --"
        $tester \
                --image-size='[299,299]' \
                --model-def=$model_def \
                --model-weights=$weights \
                $pytorch_model $mcn_model_path
    fi
}
	

mcn_model_path="${import_dir}/inception_v3-pt-mcn.mat"
convert_model "inception_v3" $mcn_model_path
