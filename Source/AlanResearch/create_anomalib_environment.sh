cd ../Anomalib/

yes | conda create -n anomalib python=3.10
source activate anomalib
pip install -e .
anomalib install
conda deactivate

cd ../AlanResearch/
