Testing the effectiveness of a simple RNN and a simple transformer encoder layer in embedding on the Penn Tree Bank
used nsight to get this 
ncu   --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum   --target-processes all   --launch-skip 20   --launch-count 20   --export AppliedMLProject1/RNNoutput.ncu-rep   -- python3 /home/wf2322/AppliedMLProject1/RNN.py
