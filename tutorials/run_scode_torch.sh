python run_scode.py --droot . \
                    --fp_exp expression_dataTuck_sub.csv \
                    --fp_trj pseudotimeTuck.txt \
                    --fp_branch cell_selectTuck.txt \
                    --num_z 10 \
                    --max_iter 10 \
                    --backend gpu \
                    --num_devices 8 \
                    --batch_size_b 100 \
                    --sp_droot out \
                    --num_repeat 6

python reconstruct_grn.py \
 --fp_rm score_result_matrix.txt \
 --fp_gn node_name.txt \
 --fp_tf mouse_tf.txt \
 --fdr 0.01 \
 --backend gpu \
 --device_ids 1