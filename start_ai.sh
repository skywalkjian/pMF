export PATH=":$PATH"
eval "$(conda shell.bash hook)"
conda activate base
cd /cpfs/user/yujian/pMF
claude --dangerously-skip-permissions
