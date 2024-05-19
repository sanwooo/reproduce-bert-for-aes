export CUDA_VISIBLE_DEVICES=1

for fold in 0 1 2 3 4
    do
	for prompt in 1 2 3 4 5 6 7 8
	    do
        python run.py --prompt_id $prompt --fold_id $fold
	done
done