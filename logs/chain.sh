python src/run_steer.py qwen 14 > logs/steer_qwen.log 2>&1
python src/run_steer.py llama 16 > logs/steer_llama.log 2>&1
python src/run_behave.py qwen gemini > logs/behave_gemini_qwen.log 2>&1
python src/run_behave.py llama gemini > logs/behave_gemini_llama.log 2>&1
python src/run_sweep.py gemma > logs/sweep_gemma.log 2>&1
python src/run_steer.py gemma 21 > logs/steer_gemma.log 2>&1
python src/run_behave.py gemma gemini > logs/behave_gemini_gemma.log 2>&1
