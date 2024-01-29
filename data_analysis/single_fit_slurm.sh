#!/bin/bash
#SBATCH --job-name=mparjo
#SBATCH --output=mparjo_%A_%a.out
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=cpu
#SBATCH --mem=128G
#


# Create a temporary Matlab preference directory

MATLAB_PREFDIR=`mktemp -d`

export MATLAB_PREFDIR


# check if that dir was created
if [[ ! "$MATLAB_PREFDIR" || ! -d "$MATLAB_PREFDIR" ]]; then
 echo "Could not cr0.eate temp preference directory: $MATLAB_PREFDIR"
 exit 1
fi


##########################
## Set-up Matlab Preferences

# You must switch off the toolbox cache in matlab.settings
# This is what it looks like in the matlab.settings XML file.
#
# <settings name="toolboxpathcache" visible="true">
# <key name="EnableToolboxPathCache" visible="true">
# <bool>
# <value>0</value>
# </bool>
# </key>
# </settings>
#
# You can create a special settings file, save it somewhere and copy it in as needed.


cp /nfs/ghome/live/hugos/Documents/MATLAB/tensor_decomp2/slurm_test/matlab.settings $MATLAB_PREFDIR


# This is also needed to disable the Toolbox path cache

cat > $MATLAB_PREFDIR/matlab.prf << EOF
#MATLAB Preferences

#Tue May 08 10:00:45 UTC 2018

RLWidthB2_WB_2014b_1_1=I130
RLWidthB2_WB_2014b_1_0=I130
CommandWindowClearConfirmation=Btrue
RLHiddenB2_WB_2014b_1=I16376
MatlabExitConfirm=Bfalse
RLPrevInitB2_WB_2014b_1=Btrue
RLOrderB2_WB_2014b_1=S0:1:
GeneralUseToolboxCache=Bfalse


EOF


##########################
## SLURM Batch starts here


hostname
srun /opt/mathworks/R2018a/bin/matlab -softwareopengl -nodisplay -nodesktop -r "data_sepi_single_fit; exit"


## SLURM Batch ends here

##########################

# deletes the temp directory
function cleanup {
 rm -rf "$MATLAB_PREFDIR"
 echo "Deleted temp preference directory: $MATLAB_PREFDIR"
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT
