param([Parameter(Mandatory=$true)][string]$Model, [Parameter(Mandatory=$true)][string]$Gsm8kFile)
python -m src.evaluation.run_eval --model $Model --tasks gsm8k --gsm8k-file $Gsm8kFile
