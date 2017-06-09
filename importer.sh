refresh_models=false
test_imported_models=true # (requires matlab.engine)

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

function convert_model()
{
    import_dir=$1
    pytorch_model=$2
    mcn_model_path="${import_dir}/${pytorch_model}-mcn.mat"
	converter="ipython $SCRIPTPATH/python/import_pytorch.py --"
	mkdir -p "${import_dir}"

    if [ $refresh_models = false ] && [ -e $mcn_model_path ]
	then
		echo "$mcn_model_path already exists; skipping."
	else
		echo "Exporting PyTorch model to matconvnet (may take some time) ..."
		$converter \
				--image-size='[224,224]' \
				--is-torchvision-model=True \
				--remove-loss \
				$pytorch_model $mcn_model_path
    fi
}

function test_model()
{
    import_dir=$1
    pytorch_model=$2
    mcn_model_path="${import_dir}/${pytorch_model}-pt-mcn.mat"
	tester="ipython $SCRIPTPATH/test/py_check.py --"
    $tester \
            --image-size='[224,224]' \
            --is-torchvision-model=True \
            $pytorch_model $mcn_model_path
}

	
import_dir="models"
#declare -a model_list=("alexnet" "vgg11" "vgg13" "vgg16" "vgg19")
declare -a model_list=("alexnet" )
declare -a im_sizes=("[227,227]" "[227 227")
for model in "${model_list[@]}"
do
    convert_model $import_dir $model
    if [ $test_imported_models = true ]
    then
        test_model $import_dir $model
    fi
done
