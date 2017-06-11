import_dir="models"
refresh_models=false
test_imported_models=false # (requires matlab.engine)

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

function convert_model()
{
    pytorch_model=$1
    mcn_model_path=$2
	converter="ipython $SCRIPTPATH/python/import_pytorch.py --"
	mkdir -p "${import_dir}"

    if [ $refresh_models = false ] && [ -e $mcn_model_path ]
	then
		echo "$mcn_model_path already exists; skipping."
	else echo "Exporting PyTorch model to matconvnet (may take some time) ..." 
    $converter \
            --image-size='[224,224]' \
            --full-image-size='[256,256]' \
            --is-torchvision-model=True \
            $pytorch_model $mcn_model_path
    fi
}

function test_model()
{
    pytorch_model=$1
    mcn_model_path=$2
	tester="ipython $SCRIPTPATH/test/py_check.py --"
    $tester \
            --image-size='[224,224]' \
            --is-torchvision-model=True \
            $pytorch_model $mcn_model_path
}

	
#declare -a model_list=("alexnet" "vgg11" "vgg13" "vgg16" "vgg19" \ 
                       #"squeezenet1_0" "squeezenet1_1")
declare -a model_list=("resnet152")
#declare -a model_list=("vgg16")
for pytorch_model in "${model_list[@]}"
do
    mcn_model_path="${import_dir}/${pytorch_model}-pt-mcn.mat"
    convert_model $pytorch_model $mcn_model_path
    if [ $test_imported_models = true ]
    then
        test_model $pytorch_model $mcn_model_path
    fi
done
