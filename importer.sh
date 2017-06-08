function convert_model()
{
	# Obtain the path of this script
	pushd `dirname $0` > /dev/null
	SCRIPTPATH=`pwd`
	popd > /dev/null

    import_dir=$1
    pytorch_model=$2
    mcn_model_path="${import_dir}/${pytorch_model}-mcn.mat"
	converter="ipython $SCRIPTPATH/python/import_pytorch.py --"
	mkdir -p "${import_dir}"

	$converter \
			--image-size='[224,224]' \
			--is-torchvision-model=True \
			--remove-dropout \
			--remove-loss \
			$pytorch_model $mcn_model_path
}
	
import_dir="models"
convert_model $import_dir "alexnet"
#convert_model $import_dir "vgg11"
