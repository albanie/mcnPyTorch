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
    model_def=$3
    weights=$4
	converter="ipython $SCRIPTPATH/python/import_pytorch.py --"
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
        tester="ipython $SCRIPTPATH/test/py_check.py --"
        $tester \
                --image-size='[224,224]' \
                --model-def=$model_def \
                --model-weights=$weights \
                $pytorch_model $mcn_model_path
    fi
}

#function test_model()
#{
    #pytorch_model=$1
    #mcn_model_path=$2
    #model_def=$3
    #weights=$4
	#tester="ipython $SCRIPTPATH/test/py_check.py --"
    #$tester \
            #--image-size='[224,224]' \
            #--model-def=$model_def \
            #--model-weights=$weights \
            #$pytorch_model $mcn_model_path
#}

	
#declare -a model_list=("alexnet" "vgg11" "vgg13" "vgg16" "vgg19" \ 
                       #"squeezenet1_0" "squeezenet1_1")
#declare -a model_list=("resnet152")
declare -a model_list=("resnext_50_32x4d")
#declare -a model_list=("vgg16")
#model_def="${HOME}/.torch/models/resnext_101_32x4d.py"
#weights="${HOME}/.torch/models/resnext_101_32x4d.pth"
model_def="${HOME}/.torch/models/resnext_50_32x4d.py"
weights="${HOME}/.torch/models/resnext_50_32x4d.pth"
for pytorch_model in "${model_list[@]}"
do
    mcn_model_path="${import_dir}/${pytorch_model}-pt-mcn.mat"
    convert_model $pytorch_model $mcn_model_path $model_def $weights
done
